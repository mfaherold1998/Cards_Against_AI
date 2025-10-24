import pandas as pd
from typing import Match
import re

pattern_id = r"(W\d{3})"
pattern_spaces = r"__+"

def get_winners_id(df:pd.DataFrame):
    
    if 'response' in df.columns:    
           
        df['winners'] = df['response'].str.findall(pattern_id)

        # If no ID is found in the response...
        mask = df['winners'].apply(lambda x: len(x) == 0)    
        df_no_response = df[mask].copy()  # All plays without answer
        
        # Remove the rows without answer from df_results
        index_to_remove = df_no_response.index
        #df_filter = df_results.drop(index_to_remove)  # create new df
        df.drop(index_to_remove, inplace=True) # use df_result

        df.drop(columns=['response'], inplace=True)

        return df_no_response
    
    #return None

def replace_with_list(iter_var, match: Match) -> str:
    try:
        return next(iter_var)
    except StopIteration:
        return match.group(0)

def build_sentence(row: pd.Series, cards: dict) -> str:

    """
    Builds the final sentence by replacing '__' in the black card
    with the texts of the winning white cards ('winners').

    Parameters:
    - row: row of df_results (with columns 'language', 'play', 'winners')
    - cards: global card dictionary, e.g., DIC_ALL_CARDS

    Returns:
    - string with the final sentence (or error message if something goes wrong)
    """

    try:
        lang = row["language"]        
        play = row["play"]
        play = play.replace("'", "").replace("[", "").replace("]", "")
        play = play.split(',')
        winners = row.get("winners", [])

        if not isinstance(play, (list, tuple)) or len(play) == 0:
            return "[BUILD_ERR: invalid play]"
        if not isinstance(winners, (list, tuple)):
            winners = [] if pd.isna(winners) else list(winners)

        black_id = play[0]
        black_text = cards[f"B_{lang}"][black_id]

        white_texts = [cards[f"W_{lang}"].get(w, f"[{w}]") for w in winners]

        n_spaces = len(re.findall(pattern_spaces, black_text))
        n_winners = len(white_texts)
        warning = ""
        if n_spaces != n_winners:
            warning = f"[WARN: {n_spaces} blank(s), {n_winners} winner(s)] "
        
        it = iter(white_texts)

        def _repl(m):
            return next(it, m.group(0))

        sentence = re.sub(pattern_spaces, _repl, black_text)

        return warning + sentence.strip()

    except KeyError as e:
        return f"[BUILD_ERR: missing card id {e}]"
    except Exception as e:
        return f"[BUILD_ERR {type(e).__name__}: {e}]"