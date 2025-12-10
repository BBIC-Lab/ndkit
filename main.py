import argparse
from omegaconf import OmegaConf

from ndkit.trainloop_runner import TrainLoopRunner
from ndkit.fit_runner import FitRunner


def get_runner(cfg):
    """
    Select runner type based on cfg.model.runner_type.
    """
    runner_type = getattr(cfg.model, "runner_type", "train_loop")

    if runner_type == "train_loop":
        return TrainLoopRunner(cfg)
    elif runner_type == "fit":
        return FitRunner(cfg)
    else:
        raise ValueError(f"Unknown runner_type: {runner_type}. "
                         f"Expected 'train_loop' or 'fit'.")


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate a neural decoder.")

    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "-m", "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "train_eval"],
        help="Execution mode.",
    )
    parser.add_argument(
        "-k", "--ckpt",
        type=str,
        default=None,
        help="Checkpoint path (used only in eval mode).",
    )

    # Parse CLI overrides
    args, unknown = parser.parse_known_args()

    # Load config + overrides
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(unknown))

    # Pick runner
    runner = get_runner(cfg)

    # Execute
    if args.mode == "train":
        runner.train()

    elif args.mode == "eval":
        runner.eval(ckpt_path=args.ckpt)

    elif args.mode == "train_eval":
        runner.train()
        runner.eval()  # Uses best checkpoint in result_dir


if __name__ == "__main__":
    main()
