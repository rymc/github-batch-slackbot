#!/bin/bash

BASE_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

CONFIG_FILE="$BASE_DIR/config.yaml"
ENV_FILE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --env)
            ENV_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--config config.yaml] [--env .env]"
            exit 1
            ;;
    esac
done

cd "$BASE_DIR"

source "$BASE_DIR/.venv/bin/activate"

pip install --quiet -r requirements.txt

CMD="python3 main.py --config \"$CONFIG_FILE\""
if [ -n "$ENV_FILE" ]; then
    CMD="$CMD --env \"$ENV_FILE\""
fi

echo "Running with config: $CONFIG_FILE"
[ -n "$ENV_FILE" ] && echo "Using env file: $ENV_FILE"
eval "$CMD" >> llm_github_bot_report.log 2>&1

deactivate 