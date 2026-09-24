import argparse
import json
from src.observability.runtime import ObservedRuntime, public_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--service")
    parser.add_argument("--environment", default="production")
    args = parser.parse_args()
    result = public_result(ObservedRuntime().invoke(vars(args)))
    print(json.dumps(result, indent=2))
    if result["status"] == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
