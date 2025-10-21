import pandas as pd
from typing import Match
import re

print("Getting winners WHITE cards IDs...")

def get_winners_id(df:pd.DataFrame, df_no_response):
    
    if 'response' in df.columns:    
        
        pattern_id = r"(W\d{3})"   
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
    
    return None

print("Building complete sentences...")

def replace_with_list(iter_var, match: Match) -> str:
    try:
        return next(iter_var)
    except StopIteration:
        return match.group(0)

def build_sentence(row, cards):
    pattern_spaces = r"__+"
    black_card_key = row.iloc[4][0]
    white_card_keys = row.iloc[5]
    row_lang = row.iloc[1]

    black_card_text = cards["B_"+row_lang][black_card_key]
    white_card_text = [cards["W_"+row_lang][key] for key in white_card_keys]
    
    iter_replace = iter(white_card_text)
    
    return re.sub(
        pattern_spaces, 
        lambda match_obj: replace_with_list(iter_replace, match_obj),
        black_card_text
    )