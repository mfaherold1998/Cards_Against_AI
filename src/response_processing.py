import pandas as pd
import re
from src.utils import convert_play_to_list

pattern_id = r"(W\d{3})"
pattern_spaces = r"__+"


def _matched_ID(row:pd.Series, cards: dict) -> bool:     
    
    play = convert_play_to_list(row["play"])
    winners = row.get("winners")        
    if not isinstance(winners, list):
        winners = [] if pd.isna(winners) else list(winners)

    try:
        black_id = play[0]
        lang = row["lang"]
        black_text = cards[f"B_{lang}"][black_id]
        if black_text is None:
             # If black card cannot be find assume a malfunction or error.
            return False
        n_spaces = len(re.findall(pattern_spaces, black_text))
        n_winners = len(winners)
        return n_spaces == n_winners

    except Exception:
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
        print("There is not 'response' column to analyze. Returning three empty DataFrames.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df_temp['winners'] = df_temp['response'].str.findall(pattern_id)

    # Split 1: responses without id
    mask_no_id = df_temp['winners'].apply(lambda x: len(x) == 0)    
    df_no_response = df_temp[mask_no_id].drop(columns=['winners']).copy()
    
    # We'll stick with the answers that do have at least one ID
    df_temp = df_temp[~mask_no_id].copy()    
    df_temp = df_temp.drop(columns=['response'])

    # Split 2: Apply the row validation function (axis=1)
    mask_matched = df_temp.apply(_matched_ID, cards=cards, axis=1)
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
    - row: row of dataframe (with columns 'language', 'play', 'winners')
    - cards: global card dictionary (e.g. DIC_ALL_CARDS).

    Returns:
    - final sentence string (or error message if something goes wrong)
    """

    try:
        lang = row["lang"]  
        play = convert_play_to_list(row["play"])
        winners = row.get("winners")        
        if not isinstance(winners, list):
            winners = [] if pd.isna(winners) else list(winners)

        black_id = play[0]
        black_text = cards[f"B_{lang}"][black_id]
        if black_text is None:
             return f"[BUILD_ERR: black card {black_id} not found for lang {lang}]"
        
        white_texts = [cards[f"W_{lang}"].get(w, f"[{w}]") for w in winners]
        # iteartor to build the sentence    
        it = iter(white_texts)

        def _repl(m):
            return next(it, m.group(0))

        sentence = re.sub(pattern_spaces, _repl, black_text)

        return sentence.strip()

    except Exception as e:
        return f"[BUILD_ERR {type(e).__name__}: {e}]"