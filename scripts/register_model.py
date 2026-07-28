import argparse
import json

from src.mlops.model_registry import ModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Register or roll back a model in MLflow Model Registry.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    register_parser = subparsers.add_parser("register")
    register_parser.add_argument("--model-uri", required=True)
    register_parser.add_argument("--description", default=None)

    latest_parser = subparsers.add_parser("latest")
    latest_parser.add_argument("--alias", default="latest")

    rollback_parser = subparsers.add_parser("rollback")
    rollback_parser.add_argument("--version", required=True)
    rollback_parser.add_argument("--alias", default="latest")

    args = parser.parse_args()
    registry = ModelRegistry()

    if args.command == "register":
        result = registry.register(args.model_uri, description=args.description)
        print(json.dumps(result.__dict__, indent=2, ensure_ascii=True))
    elif args.command == "latest":
        print(json.dumps(registry.latest(alias=args.alias), indent=2, ensure_ascii=True))
    elif args.command == "rollback":
        print(json.dumps(registry.rollback(target_version=args.version, alias=args.alias), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
