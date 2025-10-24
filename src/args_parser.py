import json
import argparse, sys

def load_config_file(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found in {filepath}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {filepath}: {e}", file=sys.stderr)
        sys.exit(1)

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config-file',
        type=str,
        required=True,
        help='Path to the JSON file with the configuration parameters.'
    )

    args = parser.parse_args()

    return load_config_file(args.config_file)
