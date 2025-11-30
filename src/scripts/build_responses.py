import pandas as pd
import re
import ast
from typing import Callable

from src.utils.logging import create_logger
logger = create_logger (log_name="main")

pattern_id = r"(W\d{3})"
pattern_spaces = r"__+"


def _match_ID_spaces(row:pd.Series, cards: dict) -> bool:     
    
    winners = row.get("winners")        
    if not isinstance(winners, list):
        winners = [] if pd.isna(winners) else list(winners)

    try:
        black_id = row["black_id"]
        lang = row["lang"]
        black_text = cards[lang]["BLACK"].loc[black_id, 'card_text']
        if black_text is None:
            logger.error(f"Black card to process not found")
            return False
        n_spaces = len(re.findall(pattern_spaces, black_text))
        n_winners = len(winners)
        return n_spaces == n_winners

    except Exception as e:
        logger.error(f"KeyError in _match_ID_spaces: {e}. Row: {row.to_dict()}")
        return False       
    
def split_responses(df:pd.DataFrame, cards: dict):
    ''' 
        Returns 3 dataframes:
        1. Filtered results of good responses (ID count matches space count).
        2. Responses without a card id in the answer.
        3. Response with mismatched ID count and space count.
    '''
    
    df_temp = df.copy()

    if 'response' not in df_temp.columns:
        logger.error("There is not 'response' column to analyze. Returning three empty DataFrames.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df_temp['winners'] = df_temp['response'].str.findall(pattern_id)

    # Split 1: responses without id
    mask_no_id = df_temp['winners'].apply(lambda x: len(x) == 0)    
    df_no_response = df_temp[mask_no_id].drop(columns=['winners']).copy()
    
    # We'll stick with the answers that do have at least one ID
    df_temp = df_temp[~mask_no_id].copy()    
    df_temp = df_temp.drop(columns=['response'])

    # Split 2: Apply the row validation function (axis=1)
    mask_matched = df_temp.apply(_match_ID_spaces, cards=cards, axis=1)
    # df_mismatch: contains the rows where the count does not match (False)
    df_mismatch = df_temp[~mask_matched].copy()

    # df_filtered: we leave the rows where the count DOES match (True)
    df_filtered = df_temp[mask_matched].copy()

    return df_filtered, df_no_response, df_mismatch

def build_sentence(row: pd.Series, cards: dict) -> str:

    """
    Builds the final sentence by replacing '__' in the black card
    with the texts of the winning white cards ('winners') considering models answers
    with an unique card id.

    Parameters:
    - row: row of dataframe (with columns 'language', 'black_id', 'winners')
    - cards: global card dictionary (e.g. DIC_ALL_CARDS).

    Returns:
    - final sentence string (or error message if something goes wrong)
    """

    try:
        lang = row["lang"]
        black_id = row["black_id"]
        winners = row.get("winners")        
        if not isinstance(winners, list):
            winners = [] if pd.isna(winners) else list(winners)

        black_text = cards[lang]["BLACK"].loc[black_id, 'card_text']
        if black_text is None:
             logger.error(f"BUILD_ERROR: Black card {black_id} not found for lang {lang}.")
             return None
        black_text = black_text.lower().strip()
        
        white_texts = [cards[lang]["WHITE"].loc[w, 'card_text'].lower().strip().rstrip('.,;!')
                       for w in winners]

        # iteartor to build the sentence    
        it = iter(white_texts)

        def _repl(m):
            return next(it, m.group(0))

        sentence = re.sub(pattern_spaces, _repl, black_text)

        return sentence.strip()

    except Exception as e:
        logger.error(f"BUILD_ERROR: {type(e).__name__}: {e}. Row: {row.to_dict()}")
        return None
    
def build_all_combinations(
    df: pd.DataFrame, 
    cards: dict, 
    build_sentence_func: Callable = build_sentence
) -> pd.DataFrame:
    """
    Expand the DataFrame to create one row for each blank card in 
    'play' and build the resulting sentence.
    """
    
    # Converting play in a list[]
    try:
        df['play_list'] = df['play'].apply(ast.literal_eval)
    except (ValueError, TypeError) as e:
        logger.error(f"Error converting play in a list: {e}")
        return pd.DataFrame()
    
    # Controlling if there is already a sentence column
    if 'sentece' in df.columns:
        df = df.drop(columns=['sentece'])

    # Expanding dataframe
    df_expanded = df.explode('play_list').copy()
    df_expanded['play_list'] = [[card_id] for card_id in df_expanded['play_list']]
    
    # Renaming columns
    df_expanded.rename(columns={'winners':'winners_of_play', 'play_list': 'winners'}, inplace=True)

    # Creating the sentences
    df_expanded['sentence'] = df_expanded.apply(build_sentence_func, axis=1, args=(cards,))

    # Renaming columns
    df_expanded.rename(columns={'winners': 'white_id', 'winners_of_play':'winners'}, inplace=True)

    # Selecting final columns
    final_cols = [
        'config', 'lang', 'model', 'temperature', 
        'winners', 'play', 'black_id', 'white_id', 'sentence'
    ]
    
    return df_expanded[final_cols]
