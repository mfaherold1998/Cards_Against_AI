from __future__ import annotations
from pathlib import Path
from typing import Dict, Literal
import pandas as pd

from src.logging import create_logger
logger = create_logger (log_name="main")

GameKey = [
    "funny_configurations_5", "funny_configurations_10",
    "random_configurations_5", "random_configurations_10",
    "toxic_configurations_5", "toxic_configurations_10",
]
CardsDict = Dict[str, Dict[str, str]]
GamesDict = Dict[str, pd.DataFrame]

REQUIRED_CARD_COLS = {"Type", "Card_Text"}

def _read_cards_safe(path: Path, file_type: Literal['xlsx', 'csv'] = 'xlsx') -> pd.DataFrame:
    """
    Read the cards file (.xlsx or .csv) in secure form. 
    The file type is determined by the 'file_type' parameter.
    """
    if not path.exists():
        logger.error(f"Cards file not found: {path}")
        raise FileNotFoundError(f"File not found: {path}") 
    
    try:
        if file_type == 'xlsx':
            df = pd.read_excel(path)
        elif file_type == 'csv':
            df = pd.read_csv(path, sep=',')
    except Exception as e:
        logger.error(f"Error reading file {path}: {e}", exc_info=False)
        raise
   
    if not REQUIRED_CARD_COLS.issubset(df.columns):
        missing_cols = REQUIRED_CARD_COLS - set(df.columns)
        logger.error(f"Missing required columns in {path.name}: {missing_cols}")
        raise ValueError(
            f"File {path} must have the columns {sorted(REQUIRED_CARD_COLS)}, "
            f"instead has {sorted(df.columns)}"
        )
    return df

def _read_games_safe(path: Path, file_type: Literal['xlsx', 'csv'] = 'xlsx') -> pd.DataFrame:

    if not path.exists():
        logger.error(f"Game file not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
    
    try:
        if file_type == 'xlsx':
            return pd.read_excel(path)
        elif file_type == 'csv':
            return pd.read_csv(path, sep=',')
        
    except Exception as e:
        logger.error(f"Error reading game file {path}: {e}", exc_info=False)
        raise

def load_cards(data_dir: Path | str, 
               langs: list[str], 
               file_type: Literal['xlsx', 'csv'] = 'xlsx') -> CardsDict:

    '''Load all BLACK and WHITE cards in all requested languages.\n
    Returns a dictionary of dictionary with all the cards texts:
    {
        "B_EN": { "B001": "text...", ... },
        "W_EN": { "W001": "text...", ... },
        "B_IT": { "B001": "text...", ... },
        "W_IT": { "W001": "text...", ... }
    }'''

    data_dir = Path(data_dir)
    DIC_ALL_CARDS: CardsDict = {}

    # Read cards text in all languages
    for lang in langs:
        path_black_cards = data_dir / lang / f"black_cards.{file_type}"
        path_white_cards = data_dir / lang / f"white_cards.{file_type}"

        try:            
            DF_BLACK = _read_cards_safe(path_black_cards, file_type=file_type)
            DF_WHITE = _read_cards_safe(path_white_cards, file_type=file_type)

            DIC_ALL_CARDS[f"B_{lang.upper()}"] = DF_BLACK.set_index("Type")["Card_Text"].to_dict()        
            DIC_ALL_CARDS[f"W_{lang.upper()}"] = DF_WHITE.set_index("Type")["Card_Text"].to_dict()
            logger.info(f"Cards in {lang.upper()} loaded...(file extension: {file_type})")
        
        except Exception as e:
            logger.error(f"Could not load cards for {lang.upper()} (Skipping language): {e}")
    
    card_set = [i for i in DIC_ALL_CARDS.keys()] 
    logger.info(f"Finished loading cards. Total loaded crads sets: {len(DIC_ALL_CARDS)} ({card_set})")
    return DIC_ALL_CARDS

def load_games(
    data_dir: Path | str,
    langs: list[str],
    dataset: Literal["all", "test"] = "test",
    subset_rows: int = 2,
    file_type: Literal['xlsx', 'csv'] = 'xlsx') -> GamesDict:

    '''Creates the dataset with all the requested games (card combinations):\n    
    If the parameter dataset==\'all\' it returns a dictionary with all the games rows.\n
    If the parameter dataset==\'test\' it returns a dictionary with the first 'subset_rows' of all configurations.\n
    If 'subset_rows' is not specified then n is 2 by default.'''

    data_dir = Path(data_dir)
    DICT_ALL_GAMES: GamesDict={}

    # Read all games configurations in all requested languages
    for lang in langs:
        for key in GameKey: 
            
            file_path = data_dir / lang / f"{key}.{file_type}"
            dict_key = f"{key}_{lang.upper()}"     

            try:
                df = _read_games_safe(file_path, file_type)
                if dataset == "test":
                    df = df.iloc[:subset_rows].copy()                    
                DICT_ALL_GAMES[dict_key] = df
                logger.info(f"Loaded: {dict_key} ({len(df)} rows)")
                
            except FileNotFoundError:
                logger.warning(f"Game file not found for {dict_key} at: {file_path}. Skipping.")
            except Exception as e:
                logger.error(f"Error reading {dict_key}. Skipping configuration. Error: {e}")       
            

    # Determine the size of the dataset
    if not DICT_ALL_GAMES:
        logger.warning("No game configurations were loaded.")
        return DICT_ALL_GAMES
    if dataset == "all":
        logger.info("Returning all loaded game rows.")
        return DICT_ALL_GAMES
    elif dataset == "test":
        logger.info("Returning subset of game rows.")
        return DICT_ALL_GAMES
    else:
        logger.error(f"Invalid value for 'dataset': {dataset}. Must be 'all' or 'test'.")
        raise ValueError(f"Invalid value for 'dataset': {dataset}. Must be 'all' or 'test'.")
    
    