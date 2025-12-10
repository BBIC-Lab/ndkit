import argparse

from omegaconf import OmegaConf
from ndkit.runner import Runner


def main():
    """Entry point for training or evaluating a neural decoding model."""
    parser = argparse.ArgumentParser(description="Train or evaluate a neural decoder.")
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "-m", "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "train_eval"],
        help="Execution mode: train / eval / train_eval.",
    )
    parser.add_argument(
        "-k", "--ckpt",
        type=str,
        default=None,
        help="Path to checkpoint for evaluation (ignored in train or train_eval mode).",
    )

    # Parse overrides, like: train.n_epochs=2 model.hidden_size=128
    args, unknown = parser.parse_known_args()

    # Load config + CLI overrides
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(unknown))

    # Initialize runner
    runner = Runner(cfg)

    # Execute according to mode
    if args.mode == "train":
        runner.train()
    elif args.mode == "eval":
        runner.eval(ckpt_path=args.ckpt)
    elif args.mode == "train_eval":
        runner.train()
        runner.eval() # Automatically evaluate using the best checkpoint

if __name__ == "__main__":
    main()
