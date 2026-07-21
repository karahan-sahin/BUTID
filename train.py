import os
import sys
import time
import math
import json
import yaml
import wandb
import torch
import datetime
import argparse

from pathlib import Path
from timm.optim import create_optimizer
from transformers import get_scheduler
from torch.utils.data import DataLoader

from src.models import UniSign, UniSignConfig, UniVQSign, UniVQSignConfig
from src.models import get_requires_grad_dict
from src.dataset import NewDataset
from src.metrics import bert_score, islr_performance_topk
from src.config import (
    train_label_paths,
    dev_label_paths,
    test_label_paths,
)
import torch.distributed as dist

sys.path.append("./third_party/unisign/")
from third_party.unisign import utils
from third_party.unisign.SLRT_metrics import translation_performance, islr_performance


def move_src_input_to_device(src_input, device, target_dtype=None):
    for key, value in src_input.items():
        if isinstance(value, torch.Tensor):
            if target_dtype is not None and torch.is_floating_point(value):
                src_input[key] = value.to(device=device, dtype=target_dtype, non_blocking=True)
            else:
                src_input[key] = value.to(device=device, non_blocking=True)
    return src_input


def main(args):
    utils.init_distributed_mode_ds(args)
    utils.set_seed(args.seed)

    print(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = output_dir.name

    if utils.is_main_process():
        wandb.init(
            project="BUTID",
            name=run_name,
            config={"args": vars(args)},
        )

    train_data = NewDataset(
        csv_path='/home/onursefa/Desktop/dataops/split/train.csv',
        pose_dir='/media/onursefa/b5fc8c03-7555-4869-804d-185d711b2219/datasets/BUTID_Mediapipe',
        args=args,
        phase='train',
        min_len=32,
        max_len=800
    )
    dev_data = NewDataset(
        csv_path='/home/onursefa/Desktop/dataops/split/val.csv',
        pose_dir='/media/onursefa/b5fc8c03-7555-4869-804d-185d711b2219/datasets/BUTID_Mediapipe',
        args=args,
        phase='train',
        min_len=32,
        max_len=800
    )
    test_data = NewDataset(
        csv_path='/home/onursefa/Desktop/dataops/split/test.csv',
        pose_dir='/media/onursefa/b5fc8c03-7555-4869-804d-185d711b2219/datasets/BUTID_Mediapipe',
        args=args,
        phase='train',
        min_len=32,
        max_len=800
    )

    print(train_data)
    print(dev_data)
    print(test_data)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_data,
        shuffle=True,
    )
    dev_sampler = torch.utils.data.distributed.DistributedSampler(
        dev_data,
        shuffle=True,
    )
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_data,
        shuffle=True,
    )

    train_num_workers = args.num_workers
    eval_num_workers = max(1, args.num_workers // 2) if args.num_workers > 0 else 0

    train_loader_kwargs = dict(
        dataset=train_data,
        batch_size=args.batch_size,
        num_workers=train_num_workers,
        collate_fn=train_data.collate_fn,
        sampler=train_sampler,
        pin_memory=False,
        drop_last=True,
        persistent_workers=False,
    )
    if train_num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = 1

    train_dataloader = DataLoader(**train_loader_kwargs)

    dev_loader_kwargs = dict(
        dataset=dev_data,
        batch_size=max(1, args.batch_size // 2),
        num_workers=eval_num_workers,
        collate_fn=dev_data.collate_fn,
        sampler=dev_sampler,
        pin_memory=False,
        persistent_workers=False,
    )
    if eval_num_workers > 0:
        dev_loader_kwargs["prefetch_factor"] = 1

    dev_dataloader = DataLoader(**dev_loader_kwargs)

    test_loader_kwargs = dict(
        dataset=test_data,
        batch_size=max(1, args.batch_size // 2),
        num_workers=eval_num_workers,
        collate_fn=test_data.collate_fn,
        sampler=test_sampler,
        pin_memory=False,
        persistent_workers=False,
    )
    if eval_num_workers > 0:
        test_loader_kwargs["prefetch_factor"] = 1

    test_dataloader = DataLoader(**test_loader_kwargs)

    if args.input_mode == "pose":
        print("Creating model...")
        config = UniSignConfig(
            modes=args.modes, 
            # proj_layer=args.proj_layer,
            proj_layer="mlp",
            hidden_dim=args.hidden_dim,
            mt5_path=args.model_name_or_path,
            label_smoothing=args.label_smoothing,
            tgt_lang="Turkish",
        )

        model = UniSign(config)
        model.train()
        
    elif args.input_mode == "vq":
        print("Creating VQ model...")
        config = UniVQSignConfig(
            modes=args.modes,
            hidden_dim=args.hidden_dim,
            mt5_path=args.model_name_or_path,
            label_smoothing=args.label_smoothing,
            part2code=args.part2code,
            tgt_lang="Turkish",
        )

        model = UniVQSign(config)
        model.train()


    for _, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.float().contiguous()

    if args.finetune != "":
        print("***********************************")
        print("Load Checkpoint...")
        print("***********************************")
        state_dict = torch.load(args.finetune, map_location="cpu")["model"]
        ret = model.load_state_dict(state_dict, strict=True)
        print("Missing keys:\n", "\n".join(ret.missing_keys))
        print("Unexpected keys:\n", "\n".join(ret.unexpected_keys))

    model_without_ddp = model
    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    print(f"number of params: {n_parameters}M")

    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(
            args.warmup_epochs * len(train_dataloader) / args.gradient_accumulation_steps
        ),
        num_training_steps=int(
            args.epochs * len(train_dataloader) / args.gradient_accumulation_steps
        ),
    )

    model, optimizer, lr_scheduler = utils.init_deepspeed(
        args, model, optimizer, lr_scheduler
    )

    # DeepSpeed engine usually exposes the wrapped model as .module
    model_without_ddp = model.module

    print(optimizer)

    start_time = time.time()
    max_accuracy = 0.0

    if args.eval:
        if utils.is_main_process():
            print("📄 dev result")
            evaluate(args, dev_dataloader, model, model_without_ddp, phase="dev")
            print("📄 test result")
            evaluate(args, test_dataloader, model, model_without_ddp, phase="test")
        return

    print(f"Start training for {args.epochs} epochs")

    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(args, model, train_dataloader, optimizer, epoch)

        if args.output_dir:
            checkpoint_path = output_dir / f"checkpoint_{epoch}.pth"
            utils.save_on_master(
                {
                    "model": get_requires_grad_dict(model_without_ddp),
                },
                checkpoint_path,
            )

        if utils.is_main_process():
            dev_stats = evaluate(
                args, dev_dataloader, model, model_without_ddp, phase="dev"
            )
            test_stats = evaluate(
                args, test_dataloader, model, model_without_ddp, phase="test"
            )

            if "SLT" in args.tasks:
                if max_accuracy < dev_stats["bleu4"]:
                    max_accuracy = dev_stats["bleu4"]
                    best_path = output_dir / "SLT" / "best_checkpoint.pth"
                    best_path.parent.mkdir(parents=True, exist_ok=True)
                    utils.save_on_master(
                        {
                            "model": get_requires_grad_dict(model_without_ddp),
                        },
                        best_path,
                    )

                print(
                    f"BLEU-4 on dev set: {dev_stats['bleu4']:.2f}"
                )
                print(f"Max BLEU-4: {max_accuracy:.2f}")

            elif "ISLR" in args.tasks:
                if max_accuracy < dev_stats["top1_acc_pi"]:
                    max_accuracy = dev_stats["top1_acc_pi"]
                    best_path = output_dir / "ISLR" / "best_checkpoint.pth"
                    best_path.parent.mkdir(parents=True, exist_ok=True)
                    utils.save_on_master(
                        {
                            "model": get_requires_grad_dict(model_without_ddp),
                        },
                        best_path,
                    )

                print(
                    f"PI accuracy on dev set: {dev_stats['top1_acc_pi']:.2f}"
                )
                print(f"Max PI accuracy: {max_accuracy:.2f}")

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"dev_{k}": v for k, v in dev_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def train_one_epoch(args, model, data_loader, optimizer, epoch):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("t_data", utils.SmoothedValue(fmt="{avg:.3f}"))
    metric_logger.add_meter("t_fwd", utils.SmoothedValue(fmt="{avg:.3f}"))
    metric_logger.add_meter("t_bwd", utils.SmoothedValue(fmt="{avg:.3f}"))
    metric_logger.add_meter("t_step", utils.SmoothedValue(fmt="{avg:.3f}"))

    header = f"Epoch: [{epoch}/{args.epochs}]"
    print_freq = 5

    optimizer.zero_grad()
    target_dtype = torch.bfloat16 if model.bfloat16_enabled() else None
    device = model.device

    _data_end = time.time()

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if batch is None or batch[0] is None:
            _data_end = time.time()
            continue

        src_input, tgt_input = batch
        metric_logger.update(t_data=time.time() - _data_end)

        src_input = move_src_input_to_device(src_input, device, target_dtype=target_dtype)

        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        _fwd_start = time.time()

        stack_out = model(src_input, tgt_input)

        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        metric_logger.update(t_fwd=time.time() - _fwd_start)

        total_loss = stack_out["loss"]

        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        _bwd_start = time.time()

        model.backward(total_loss)

        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        metric_logger.update(t_bwd=time.time() - _bwd_start)

        _step_start = time.time()
        model.step()

        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        metric_logger.update(t_step=time.time() - _step_start)

        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        for k, v in stack_out.items():
            if "loss" in k and v is not None:
                metric_logger.update(**{k: v.item()})

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        _data_end = time.time()

        if utils.is_main_process():
            wandb.log(
            {
                    **{
                        f"train/{k}": v.item()
                        for k, v in stack_out.items()
                        if "loss" in k and v is not None
                    },
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
            )

        del stack_out, total_loss

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(args, data_loader, model, model_without_ddp, phase):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"{phase.upper()}:"

    target_dtype = torch.bfloat16 if model.bfloat16_enabled() else None
    device = model.device
    tokenizer = model_without_ddp.mt5_tokenizer

    eval_loss = 0.0
    tgt_pres, tgt_refs, tasks = [], [], []
    max_eval_steps = getattr(args, "max_eval_steps", None)
    skip_on_error = True

    main_print = print if utils.is_main_process() else lambda *a, **k: None

    with torch.no_grad():
        data_iter = iter(data_loader)
        step = 0

        for _ in metric_logger.log_every(range(len(data_loader)), 10, header):

            if max_eval_steps is not None and step >= max_eval_steps:
                main_print(f"[Eval] Reached max_eval_steps={max_eval_steps}, stopping early.")
                break

            # ---- DataLoader safety ----
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            except Exception as e:
                main_print(f"[Eval][Loader][Step {step}] DataLoader error: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                step += 1
                continue

            if batch is None:
                main_print(f"[Eval][Step {step}] Skipping None batch")
                step += 1
                continue

            try:
                if batch[0] is None:
                    main_print(f"[Eval][Step {step}] Skipping invalid batch[0]")
                    step += 1
                    continue

                src_input, tgt_input = batch
                src_input = move_src_input_to_device(
                    src_input, device, target_dtype=target_dtype
                )

                stack_out = model(src_input, tgt_input)

                if "loss" not in stack_out:
                    main_print(f"[Eval][Step {step}] Missing loss in output, skipping")
                    step += 1
                    continue

                total_loss = stack_out["loss"]

                if not torch.isfinite(total_loss):
                    main_print(f"[Eval][Step {step}] Non-finite loss, skipping")
                    step += 1
                    continue

                loss_item = total_loss.item()
                metric_logger.update(loss=loss_item)
                eval_loss += loss_item

                # ---- generation safety ----
                try:
                    output = model_without_ddp.generate(
                        stack_out,
                        max_new_tokens=getattr(args, "eval_max_tokens", 100),
                        num_beams=getattr(args, "eval_beams", 4),
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        main_print(f"[Eval][Step {step}] OOM during generate, skipping batch")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        step += 1
                        continue
                    raise

                pred_texts = tokenizer.batch_decode(
                    output.detach().cpu(), skip_special_tokens=True
                )

                labels = tgt_input["labels"]
                if isinstance(labels, torch.Tensor):
                    labels = labels.detach().cpu().clone()
                    labels[labels == -100] = tokenizer.pad_token_id
                    ref_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
                else:
                    ref_texts = [str(x) for x in labels]

                tgt_pres.extend(pred_texts)
                tgt_refs.extend(ref_texts)
                tasks.extend(src_input.get("tasks", ["unknown"] * len(pred_texts)))

                del stack_out, total_loss, output

            except Exception as e:
                main_print(f"[Eval][Step {step}] Batch error: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if not skip_on_error:
                    raise

            step += 1

    if utils.is_main_process():
        wandb.log(
            {
                f"{phase}/loss": eval_loss / max(1, len(data_loader)),
            }
        )

    if args.dataset == "CSL_Daily":
        tgt_pres = [
            " ".join(list(r.replace(" ", "").replace("\n", ""))) if task == "SLT" else r
            for r, task in zip(tgt_pres, tasks)
        ]
        tgt_refs = [
            " ".join(list(r.replace("，", ",").replace("？", "?").replace(" ", "")))
            if task == "SLT"
            else r
            for r, task in zip(tgt_refs, tasks)
        ]

    if "SLT" in args.tasks:
        bleu_dict, rouge_score = translation_performance(tgt_refs, tgt_pres)

        for k, v in bleu_dict.items():
            metric_logger.meters[k].update(v)
        metric_logger.meters["rouge"].update(rouge_score)

        # print samples
        max_log = args.num_log_samples if hasattr(args, "num_log_samples") else 20
        print(f"\n--- Sample Predictions ({phase}) ---")
        for i in range(min(max_log, len(tgt_pres))):
            print(f"Ref: {tgt_refs[i]}")
            print(f"Pred: {tgt_pres[i]}")
            print("---")

        # add later
        # bert_scores = bert_score(tgt_pres, tgt_refs, lang="tr")
        # metric_logger.meters["bert_score"].update(bert_scores["f1"])

        print(
            f"## SLT results ##\n"
            f"BLEU-1: {bleu_dict['bleu1']:.2f}%\n"
            f"BLEU-2: {bleu_dict['bleu2']:.2f}%\n"
            f"BLEU-3: {bleu_dict['bleu3']:.2f}%\n"
            f"BLEU-4: {bleu_dict['bleu4']:.2f}%\n"
            f"ROUGE: {rouge_score:.2f}%\n"
            # f"BERT Score: {bert_scores['f1']:.2f}%\n"
        )

        if utils.is_main_process():
            wandb.log(
                {
                    f"{phase}/bleu1": bleu_dict["bleu1"],
                    f"{phase}/bleu2": bleu_dict["bleu2"],
                    f"{phase}/bleu3": bleu_dict["bleu3"],
                    f"{phase}/bleu4": bleu_dict["bleu4"],
                    f"{phase}/rouge": rouge_score,
                    # f"{phase}/bert_score": bert_scores["f1"],
                }
            )

    if "ISLR" in args.tasks:
        top1_acc_pi, top1_acc_pc = islr_performance(tgt_refs, tgt_pres)
        metric_logger.meters["top1_acc_pi"].update(top1_acc_pi)
        metric_logger.meters["top1_acc_pc"].update(top1_acc_pc)

        print(
            f"ISLR results\n"
            f"Top-1 Acc (P-I): {top1_acc_pi:.2f}%\n"
            f"Top-1 Acc (P-C): {top1_acc_pc:.2f}%"
        )

        if utils.is_main_process():
            wandb.log(
                {
                    f"{phase}/top1_acc_pi": top1_acc_pi,
                    f"{phase}/top1_acc_pc": top1_acc_pc,
                }
            )

    if utils.is_main_process() and utils.get_world_size() == 1 and args.eval:
        with open(args.output_dir + f"/{phase}_tmp_pres.txt", "w") as f:
            for pred in tgt_pres:
                f.write(pred + "\n")
        with open(args.output_dir + f"/{phase}_tmp_refs.txt", "w") as f:
            for ref in tgt_refs:
                f.write(ref + "\n")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from src.args import get_args_parser

    parser = argparse.ArgumentParser("Uni-Sign scripts", parents=[get_args_parser()])

    parser.add_argument("--config", default="", type=str, help="path to a YAML config file")
    parser.add_argument(
        "--online_data_loading",
        action="store_true",
        help="whether to load pose and vq features on-the-fly during training/evaluation; set to false to pre-extract and save all features to disk before training (not recommended due to large storage requirements)",
    )
    parser.add_argument(
        "--start_pad",
        default=3,
        type=int,
        help="seconds before the start timestamp of a sample to include in the pose and vq features",
    )
    parser.add_argument(
        "--end_pad",
        default=3,
        type=int,
        help="seconds after the end timestamp of a sample to include in the pose and vq features",
    )
    parser.add_argument(
        "--max_open_h5",
        default=8,
        type=int,
        help="maximum number of lazily opened h5 files to keep per worker",
    )
    parser.add_argument(
        "--input_mode",
        default="pose",
        type=str,
        choices=["pose", "vq"],
        help="input mode for the model",
    )


    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)
        args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
