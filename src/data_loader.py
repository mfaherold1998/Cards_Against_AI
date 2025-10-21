from __future__ import annotations
from pathlib import Path
from typing import Dict, Literal, Optional
import pandas as pd

CardsDict = Dict[str, Dict[str, str]]
GameKey = Literal[
    "FUNNY_5_EN", "FUNNY_10_EN",
    "RANDOM_5_EN", "RANDOM_10_EN",
    "TOXIC_5_EN", "TOXIC_10_EN",
]
GamesDict = Dict[GameKey, pd.DataFrame]

REQUIRED_CARD_COLS = {"Type", "Card_Text"}

def _read_cards_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_excel(path)
    if not REQUIRED_CARD_COLS.issubset(df.columns):
        raise ValueError(
            f"File {path} must have the columns {sorted(REQUIRED_CARD_COLS)}, "
            f"instead has {sorted(df.columns)}"
        )
    return df

def load_cards(data_dir: Path | str = "./cards_dataset") -> CardsDict:

    ''' Load all BLACK and WHITE cards in all available languages. Currently: UK English.\n
    Returns a dictionary of dictionary with all the cards:
    {
        "B_EN": { "B001": "text...", ... },
        "W_EN": { "W001": "text...", ... }
    }'''

    data_dir = Path(data_dir)

    # Cards text
    DF_BLACK_CARDS_EN=_read_cards_safe(data_dir / "EN/BLACK_cards.xlsx")
    DF_WHITE_CARDS_EN=_read_cards_safe(data_dir / "EN/WHITE_cards.xlsx")

    # Cards Dictionary #TYPEDEF
    DIC_ALL_CARDS: CardsDict={
                    "B_EN": DF_BLACK_CARDS_EN.set_index("Type")["Card_Text"].to_dict().copy(),
                    "W_EN": DF_WHITE_CARDS_EN.set_index("Type")["Card_Text"].to_dict().copy(),
                    "B_IT": {},
                    "W_IT": {},
                    "B_ES": {},
                    "W_ES": {}
                }
    
    return DIC_ALL_CARDS


def config_games(
    dataset: Literal["all", "test"] = "test",
    data_dir: Path | str = "./cards_dataset",
    subset_n: Optional[int] = 2,) -> GamesDict:

    '''Creates the dataset with all the available games (card combinations):\n    
    If the parameter dataset==\'all\' it returns a dictionary with all the games.\n
    If the parameter dataset==\'test\' it returns a dictionary with the first n rows of all the games.\n
    If subset_n is not specified then n is 2 by default.'''

    data_dir = Path(data_dir)

    # All Games
    DF_FUNNY_5_EN=pd.read_excel(data_dir / "EN/funny_configurations_5.xlsx")
    #DF_FUNNY_10_EN=pd.read_excel(data_dir / "EN/funny_configurations_10.xlsx")
    DF_RANDOM_5_EN=pd.read_excel(data_dir / "EN/random_configurations_5.xlsx")
    #DF_RANDOM_10_EN=pd.read_excel(data_dir / "EN/random_configurations_10.xlsx")
    DF_TOXIC_5_EN=pd.read_excel(data_dir / "EN/toxic_configurations_ID_5.xlsx")
    #DF_TOXIC_10_EN=pd.read_excel(data_dir / "EN/toxic_configurations_ID_10.xlsx")

    DICT_ALL_GAMES: GamesDict={
        'FUNNY_5_EN': DF_FUNNY_5_EN.copy(),
        #'FUNNY_10_EN': DF_FUNNY_10_EN.copy(),
        'RANDOM_5_EN': DF_RANDOM_5_EN.copy(),
        #'RANDOM_10_EN': DF_RANDOM_10_EN.copy(),
        'TOXIC_5_EN': DF_TOXIC_5_EN.copy()
        #'TOXIC_10_EN': DF_TOXIC_10_EN.copy()
    }

    if dataset == "all":
        return DICT_ALL_GAMES
    
    n = 2 if subset_n is None else int(subset_n)
    return {k: v.iloc[:n].copy() for k, v in DICT_ALL_GAMES.items()}
    