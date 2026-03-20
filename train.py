import wandb
import os
import time
import argparse, json, datetime, yaml
from pathlib import Path
import math
import sys
from timm.optim import create_optimizer
from transformers import get_scheduler

from src.models import UniSign, UniSignConfig
from src.models import get_requires_grad_dict

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Add unisign path to system environment

from src.dataset import BUTIDDataset
from src.metrics import bert_score, islr_performance_topk
from src.config import (
    train_label_paths,
    dev_label_paths,
    test_label_paths,
)

sys.path.append("./third_party/unisign/")
from third_party.unisign import utils
from third_party.unisign.SLRT_metrics import translation_performance, islr_performance


def main(args):

    utils.init_distributed_mode_ds(args)

    print(args)

    utils.set_seed(args.seed)

    run_name = args.output_dir.split("/")[-1]

    if utils.is_main_process():
        wandb.init(
            project="BUTID",
            name=run_name,
            config={"args": vars(args)},
        )

    train_data = BUTIDDataset(
        path=train_label_paths[args.dataset], args=args, phase="train"
    )
    print(train_data)
    print(train_data[5000])

    dev_data = BUTIDDataset(path=dev_label_paths[args.dataset], args=args, phase="dev")
    print(dev_data)
    test_data = BUTIDDataset(
        path=test_label_paths[args.dataset], args=args, phase="test"
    )
    print(test_data)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_data, shuffle=True
    )
    # train_sampler = torch.utils.data.RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=train_data.collate_fn,
        sampler=train_sampler,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_data,shuffle=False)
    dev_sampler = torch.utils.data.SequentialSampler(dev_data)
    dev_dataloader = DataLoader(
        dev_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dev_data.collate_fn,
        sampler=dev_sampler,
        pin_memory=args.pin_mem,
    )

    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_data,shuffle=False)
    test_sampler = torch.utils.data.SequentialSampler(test_data)
    test_dataloader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=test_data.collate_fn,
        sampler=test_sampler,
        pin_memory=args.pin_mem,
    )

    print(f"Creating model:")
    config = UniSignConfig(
        modes=args.modes,
        hidden_dim=args.hidden_dim,
        mt5_path=args.model_name_or_path,
        label_smoothing=args.label_smoothing,
        tgt_lang="Turkish",
    )
    model = UniSign(config)
    model.cuda()
    model.train()
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    if args.finetune != "":
        print("***********************************")
        print("Load Checkpoint...")
        print("***********************************")
        state_dict = torch.load(args.finetune, map_location="cpu")["model"]

        ret = model.load_state_dict(state_dict, strict=True)
        print("Missing keys: \n", "\n".join(ret.missing_keys))
        print("Unexpected keys: \n", "\n".join(ret.unexpected_keys))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module
    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    print(f"number of params: {n_parameters}M")

    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(
            args.warmup_epochs
            * len(train_dataloader)
            / args.gradient_accumulation_steps
        ),
        num_training_steps=int(
            args.epochs * len(train_dataloader) / args.gradient_accumulation_steps
        ),
    )

    model, optimizer, lr_scheduler = utils.init_deepspeed(
        args, model, optimizer, lr_scheduler
    )
    model_without_ddp = model.module.module

    print(optimizer)

    output_dir = Path(args.output_dir)

    start_time = time.time()
    max_accuracy = 0
    if args.eval:
        if utils.is_main_process():
            print("📄 dev result")
            evaluate(args, dev_dataloader, model, model_without_ddp, phase="dev")
            print("📄 test result")
            evaluate(args, test_dataloader, model, model_without_ddp, phase="test")

        return
    print(f"Start training for {args.epochs} epochs")

    for epoch in range(0, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(args, model, train_dataloader, optimizer, epoch)

        if args.output_dir:
            checkpoint_paths = [output_dir / f"checkpoint_{epoch}.pth"]
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model": get_requires_grad_dict(model_without_ddp),
                    },
                    checkpoint_path,
                )

        # single gpu inference
        if utils.is_main_process():
            test_stats = evaluate(
                args, dev_dataloader, model, model_without_ddp, phase="dev"
            )
            evaluate(args, test_dataloader, model, model_without_ddp, phase="test")

            if "SLT" in args.tasks:
                if max_accuracy < test_stats["bleu4"]:
                    max_accuracy = test_stats["bleu4"]
                    if args.output_dir and utils.is_main_process():
                        checkpoint_paths = [output_dir / "SLT" / "best_checkpoint.pth"]
                        for checkpoint_path in checkpoint_paths:
                            utils.save_on_master(
                                {
                                    "model": get_requires_grad_dict(model_without_ddp),
                                },
                                checkpoint_path,
                            )

                print(
                    f"BLEU-4 of the network on the {len(dev_dataloader)} dev videos: {test_stats['bleu4']:.2f}"
                )
                print(f"Max BLEU-4: {max_accuracy:.2f}%")

            elif "ISLR" in args.tasks:
                if max_accuracy < test_stats["top1_acc_pi"]:
                    max_accuracy = test_stats["top1_acc_pi"]
                    if args.output_dir and utils.is_main_process():
                        checkpoint_paths = [output_dir / "ISLR" / "best_checkpoint.pth"]
                        for checkpoint_path in checkpoint_paths:
                            utils.save_on_master(
                                {
                                    "model": get_requires_grad_dict(model_without_ddp),
                                },
                                checkpoint_path,
                            )

                print(
                    f"PI accuracy of the network on the {len(dev_dataloader)} dev videos: {test_stats['top1_acc_pi']:.2f}"
                )
                print(f"Max PI accuracy: {max_accuracy:.2f}%")

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def train_one_epoch(args, model, data_loader, optimizer, epoch):
    model.train()

    from deepspeed.profiling.flops_profiler import FlopsProfiler
    profile_step = len(data_loader) // 2
    prof = FlopsProfiler(model)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr",     utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("t_data", utils.SmoothedValue(fmt="{avg:.3f}"))
    metric_logger.add_meter("t_fwd",  utils.SmoothedValue(fmt="{avg:.3f}"))
    metric_logger.add_meter("t_bwd",  utils.SmoothedValue(fmt="{avg:.3f}"))
    metric_logger.add_meter("t_step", utils.SmoothedValue(fmt="{avg:.3f}"))

    header = f"Epoch: [{epoch}/{args.epochs}]"
    print_freq = 5

    optimizer.zero_grad()
    target_dtype = torch.bfloat16 if model.bfloat16_enabled() else None

    _data_end = time.time()
    for step, (src_input, tgt_input) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        metric_logger.update(t_data=time.time() - _data_end)

        if step == profile_step:
            prof.start_profile()

        if target_dtype != None:
            for key in src_input.keys():
                if isinstance(src_input[key], torch.Tensor):
                    src_input[key] = src_input[key].to(target_dtype).cuda()

        torch.cuda.synchronize()
        _fwd_start = time.time()
        stack_out = model(src_input, tgt_input)
        torch.cuda.synchronize()
        metric_logger.update(t_fwd=time.time() - _fwd_start)

        total_loss = stack_out["loss"]
        torch.cuda.synchronize()
        _bwd_start = time.time()
        model.backward(total_loss)
        torch.cuda.synchronize()
        metric_logger.update(t_bwd=time.time() - _bwd_start)

        _step_start = time.time()
        model.step()
        torch.cuda.synchronize()
        metric_logger.update(t_step=time.time() - _step_start)

        if step == profile_step and args.debug:
            prof.print_model_profile(profile_step=profile_step)
            prof.end_profile()

        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        for k in stack_out:
            if "loss" in k and stack_out[k] is not None:
                metric_logger.update(**{k: stack_out[k].item()})

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        _data_end = time.time()

        if utils.is_main_process():
            wandb.log({
                **{
                f"train/{k}": stack_out[k].item()
                for k in stack_out
                if "loss" in k and stack_out[k] is not None
                },
                "train/lr": optimizer.param_groups[0]["lr"],
            })
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(args, data_loader, model, model_without_ddp, phase):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    target_dtype = None
    if model.bfloat16_enabled():
        target_dtype = torch.bfloat16

    eval_loss = 0
    with torch.no_grad():
        tgt_pres, tgt_refs, tasks = [], [], []
        for step, (src_input, tgt_input) in enumerate(
            metric_logger.log_every(data_loader, 10, header)
        ):
            if target_dtype != None:
                for key in src_input.keys():
                    if isinstance(src_input[key], torch.Tensor):
                        src_input[key] = src_input[key].to(target_dtype).cuda()

            stack_out = model(src_input, tgt_input)

            total_loss = stack_out["loss"]
            metric_logger.update(loss=total_loss.item())
            eval_loss += total_loss.item()

            output = model_without_ddp.generate(
                stack_out, max_new_tokens=100, num_beams=4
            )

            for i in range(len(output)):
                tgt_pres.append(output[i])
                tgt_refs.append(tgt_input["labels"][i])
                tasks.append(src_input["tasks"][i])

    wandb.log(
        {
            f"{phase}/loss": eval_loss / len(data_loader),
        }
    )

    tokenizer = model_without_ddp.mt5_tokenizer
    padding_value = tokenizer.eos_token_id

    pad_tensor = torch.ones(150 - len(tgt_pres[0])).cuda() * padding_value
    tgt_pres[0] = torch.cat((tgt_pres[0], pad_tensor.long()), dim=0)

    tgt_pres = pad_sequence(tgt_pres, batch_first=True, padding_value=padding_value)
    tgt_pres = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)

    # fix mt5 tokenizer bug
    if args.dataset == "CSL_Daily":
        tgt_pres = [
            " ".join(list(r.replace(" ", "").replace("\n", ""))) if task == "SLT" else r
            for r, task in zip(tgt_pres, tasks)
        ]
        tgt_refs = [
            (
                " ".join(list(r.replace("，", ",").replace("？", "?").replace(" ", "")))
                if task == "SLT"
                else r
            )
            for r, task in zip(tgt_refs, tasks)
        ]

    if "SLT" in args.tasks:
        bleu_dict, rouge_score = translation_performance(tgt_refs, tgt_pres)
        bert_scores = bert_score(tgt_pres, tgt_refs, lang="tr")
        for k, v in bleu_dict.items():
            metric_logger.meters[k].update(v)
        metric_logger.meters["rouge"].update(rouge_score)
        metric_logger.meters["bert_score"].update(bert_scores["f1"])

        print(
            f"## SLT results  ##\n"
            f"BLEU-1: {bleu_dict['bleu1']:.2f}%\n"
            f"BLEU-2: {bleu_dict['bleu2']:.2f}%\n"
            f"BLEU-3: {bleu_dict['bleu3']:.2f}%\n"
            f"BLEU-4: {bleu_dict['bleu4']:.2f}%\n"
            f"ROUGE: {rouge_score:.2f}%\n"
            f"BERT Score: {bert_scores['f1']:.2f}%\n"
        )
        wandb.log(
            {
                f"{phase}/bleu1": bleu_dict["bleu1"],
                f"{phase}/bleu2": bleu_dict["bleu2"],
                f"{phase}/bleu3": bleu_dict["bleu3"],
                f"{phase}/bleu4": bleu_dict["bleu4"],
                f"{phase}/rouge": rouge_score,
                f"{phase}/bert_score": bert_scores["f1"],
            }
        )

    if "ISLR" in args.tasks:
        top1_acc_pi, top1_acc_pc = islr_performance(tgt_refs, tgt_pres)
        metric_logger.meters["top1_acc_pi"].update(top1_acc_pi)
        metric_logger.meters["top1_acc_pc"].update(top1_acc_pc)

        # top5_acc_pi, top5_acc_pc = islr_performance_topk(tgt_refs, tgt_pres, k=5)
        # metric_logger.meters['top5_acc_pi'].update(top5_acc_pi)
        # metric_logger.meters['top5_acc_pc'].update(top5_acc_pc)
        print(
            f"ISLR results\n"
            f"Top-1 Acc (P-I): {top1_acc_pi:.2f}%\n"
            # f"Top-5 Acc (PI): {top5_acc_pi:.2f}%\n"
            # '#############################\n'
            f"Top-1 Acc (P-C): {top1_acc_pc:.2f}%"
            # f"Top-5 Acc (PC): {top5_acc_pc:.2f}%\n"
        )
        wandb.log(
            {
                f"{phase}/top1_acc_pi": top1_acc_pi,
                f"{phase}/top1_acc_pc": top1_acc_pc,
                # f'{phase}/top5_acc_pi': top5_acc_pi,
                # f'{phase}/top5_acc_pc': top5_acc_pc,
            }
        )

    # # gather the stats from all processes
    # metric_logger.synchronize_between_processes()

    if utils.is_main_process() and utils.get_world_size() == 1 and args.eval:
        with open(args.output_dir + f"/{phase}_tmp_pres.txt", "w") as f:
            for i in range(len(tgt_pres)):
                f.write(tgt_pres[i] + "\n")
        with open(args.output_dir + f"/{phase}_tmp_refs.txt", "w") as f:
            for i in range(len(tgt_refs)):
                f.write(tgt_refs[i] + "\n")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from src.args import get_args_parser

    parser = argparse.ArgumentParser("Uni-Sign scripts", parents=[get_args_parser()])

    parser.add_argument(
        "--config", default="", type=str, help="path to a YAML config file"
    )
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        # Set YAML values as defaults so CLI args take precedence
        parser.set_defaults(**cfg)
        args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
