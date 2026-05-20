import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="auto-trojan", description="LLM red-teaming framework")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run a red-team campaign")
    run_parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    run_parser.add_argument("--seed-prompts", nargs="+", help="Override seed prompts from config")

    args = parser.parse_args()

    if args.command == "run":
        from omegaconf import OmegaConf  # type: ignore[import]
        cfg = OmegaConf.load(args.config)
        print(f"Loaded config: {args.config}")
        print(OmegaConf.to_yaml(cfg))
        print("Campaign runner not yet implemented — wire up RLLoop here.")
    else:
        parser.print_help()
        sys.exit(1)
