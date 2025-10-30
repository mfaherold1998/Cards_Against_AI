from __future__ import annotations
from pathlib import Path
from typing import Dict, Literal
import pandas as pd

GameKey = [
    "funny_configurations_5", "funny_configurations_10",
    "random_configurations_5", "random_configurations_10",
    "toxic_configurations_5", "toxic_configurations_10",
]
CardsDict = Dict[str, Dict[str, str]]
GamesDict = Dict[GameKey, pd.DataFrame]

REQUIRED_CARD_COLS = {"Type", "Card_Text"}

def _read_cards_safe(path: Path, file_type: Literal['xlsx', 'csv'] = 'xlsx') -> pd.DataFrame:
    """
    Read the cards file (.xlsx or .csv) in secure form. 
    The file type is determined by the 'file_type' parameter.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if file_type == 'xlsx':
        df = pd.read_excel(path)
    elif file_type == 'csv':
        df = pd.read_csv(path, sep=',')
    else:
        raise ValueError("file_type must be 'xlsx' or 'csv'")
    if not REQUIRED_CARD_COLS.issubset(df.columns):
        raise ValueError(
            f"File {path} must have the columns {sorted(REQUIRED_CARD_COLS)}, "
            f"instead has {sorted(df.columns)}"
        )
    return df

def _read_games_safe(path: Path, file_type: Literal['xlsx', 'csv'] = 'xlsx') -> pd.DataFrame:
    
    if file_type == 'xlsx':
        return pd.read_excel(path)
    elif file_type == 'csv':
        return pd.read_csv(path, sep=',')
    else:
        raise ValueError("Error: file_type must be 'xlsx' or 'csv'")

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
            print(f"Cards in {lang.upper()} loaded...(file extension: {file_type})")
        
        except Exception as e:
            print(f"The cards could not be loaded for {lang.upper()}: {e}")
    
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
                DICT_ALL_GAMES[dict_key] = df
                print(f"Loaded: {dict_key} ({len(df)} rows)")
                
            except FileNotFoundError:
                print(f"File not found for {dict_key} at: {file_path}")
            except Exception as e:
                print(f"Error reading {dict_key}: {e}")       
            

    # Determine the size of the dataset
    if not DICT_ALL_GAMES or dataset == "all":
        return DICT_ALL_GAMES
    elif dataset == "test":
        print(f"Returning the first {subset_rows} rows of each configuration.")
        return {k: v.iloc[:subset_rows].copy() for k, v in DICT_ALL_GAMES.items()}
    
    raise ValueError(f"Invalid value for 'dataset': {dataset}. Must be 'all' or 'test'.")
