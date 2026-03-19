import argparse


def get_args_parser():
    parser = argparse.ArgumentParser("Uni-Sign scripts", add_help=False)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--gradient-accumulation-steps", default=8, type=int)
    parser.add_argument("--gradient-clipping", default=1.0, type=float)
    parser.add_argument("--epochs", default=20, type=int)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)

    # * Finetuning params
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")

    # * Optimizer parameters
    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw"',
    )
    parser.add_argument(
        "--opt-eps",
        default=1.0e-09,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1.0e-09)",
    )
    parser.add_argument(
        "--opt-betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: [0.9, 0.98], use opt default)",
    )
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
        help="weight decay (default: 0.05)",
    )

    parser.add_argument(
        "--sched",
        default="cosine",
        type=str,
        metavar="SCHEDULER",
        help='LR scheduler (default: "cosine"',
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0e-3,
        metavar="LR",
        help="learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1.0e-08,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=float,
        default=0,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )

    # * Baise params
    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument(
        "--pin-mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem", help="")
    parser.set_defaults(pin_mem=True)

    # deepspeed features
    parser.add_argument(
        "--offload", action="store_true", help="Enable ZeRO Offload techniques."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp16", "bf16"],
        help="Training data type",
    )
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=2,
        help="ZeRO optimization stage for Actor model (and clones).",
    )
    ## low precision
    parser.add_argument(
        "--compute_fp32_loss",
        action="store_true",
        help="Relevant for low precision dtypes (fp16, bf16, etc.). "
        "If specified, loss is calculated in fp32.",
    )

    parser.add_argument(
        "--quick_break", type=int, default=0, help="save ckpt per quick_break step"
    )

    # RGB branch
    parser.add_argument(
        "--rgb_support",
        action="store_true",
    )

    # Pose length
    parser.add_argument("--max_length", default=256, type=int)

    # select dataset
    parser.add_argument(
        "--dataset",
        default="BUTID",
        choices=["BUTID", "BSign22k", "AUTSL"],
    )

    # select task
    parser.add_argument("--task", default="SLT", choices=["SLT", "ISLR", "CSLR"])

    # select label smooth
    parser.add_argument("--label_smoothing", default=0.2, type=float)

    # online inference
    parser.add_argument("--online_video", default="", type=str)

    parser.add_argument(
        "--modes",
        nargs="+",
        default=["body", "right", "left"],
        type=str,
        help="input modes: body, right, left, face",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="google/mt5-small",
        type=str,
        help="pretrained model name or path for mt5",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=["SLT"], type=str, help="tasks: ISLR, SLT"
    )
    parser.add_argument(
        "--print_freq", default=10, type=int, help="print frequency during training"
    )
    parser.add_argument(    
        "--debug", action="store_true", help="Run in debug mode with fewer epochs and smaller dataset"
    )
    parser.add_argument(
        "--input_type",
        default="pose",
        choices=["pose", "vq"],
        help="Type of input data (pose or vq)",
    )

    return parser
