import json
import argparse, sys
from src.data.schema import (
    JsonConfigSchema_1, 
    JsonConfigSchema_2, 
    JsonConfigSchema_3,
    JsonConfigSchema_5
    )
from pydantic import ValidationError

from src.utils.logging import create_logger
logger = create_logger (log_name="main")

SCHEMA_NUM_MAP = {
    1 : JsonConfigSchema_1,
    2 : JsonConfigSchema_2,
    3: JsonConfigSchema_3,
    4: JsonConfigSchema_2, # File erased!!
    5: JsonConfigSchema_5
}

def load_config_file(filepath: str, num_file: int):
    try:
        
        jfile = {}
        schema = SCHEMA_NUM_MAP[num_file]
        
        with open(filepath, 'r') as f:
            jfile = json.load(f)

        schema(**jfile)
        return jfile

    except FileNotFoundError:
        logger.critical(f"FATAL: Configuration file not found at {filepath}. Exiting.", exc_info=False)
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.critical(f"FATAL: Error decoding JSON in {filepath}. Error: {e}", exc_info=True)
        sys.exit(1)
    except ValidationError as e:
        logger.critical(f"FATAL: Configuration failed schema validation in {filepath}. Error details: {e}", exc_info=False)
        sys.exit(1)
    except KeyError:
        logger.critical(f"FATAL: Schema number {num_file} not found in SCHEMA_NUM_MAP. Exiting.", exc_info=False)
        sys.exit(1)

def get_args(num_file: int):

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config-file',
        type=str,
        required=True,
        help='Path to the JSON file with the configuration parameters.'
    )

    args = parser.parse_args()

    return load_config_file(args.config_file, num_file)  # args.config_file = '*_config.json'
