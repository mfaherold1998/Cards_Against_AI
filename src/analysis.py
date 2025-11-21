import pandas as pd
import ast
from collections import Counter
from typing import Dict, Any

from src.logging import create_logger
logger = create_logger (log_name="main")

def frequencies_election(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the frequency of selection for each white card by dividing 
    the number of times it won by the number of times it was available.
    """
    
    df_temp = df.copy()

    # Convert columns to list
    try:
        df_temp['play_list'] = df_temp['play'].apply(ast.literal_eval)
        df_temp['winners_list'] = df_temp['winners'].apply(ast.literal_eval)
    except (ValueError, TypeError) as e:
        logger.error(f"Error coverting to list column play or winners: {e}", exc_info=True)
        return pd.DataFrame() # Return empty df in case of error
    
    df_temp = df_temp[['play_list', 'winners_list']]

    # Counting cards frequencies
    all_availables = [card for sublist in df_temp['play_list'] for card in sublist]
    counting_availables = Counter(all_availables)

    # Counting winners frequencies
    todas_ganadoras = [card for sublist in df_temp['winners_list'] for card in sublist]
    conteo_ganadora = Counter(todas_ganadoras)

    # Getting unique white cards
    all_cards = set(all_availables)

    # Results
    results = pd.DataFrame({
        'card_id': list(all_cards)
    })

    results['available_freq'] = results['card_id'].apply(lambda x: counting_availables.get(x, 0))
    results['winner_freq'] = results['card_id'].apply(lambda x: conteo_ganadora.get(x, 0))

    results['election_freq'] = results.apply(
        lambda row: row['winner_freq'] / row['available_freq']
        if row['available_freq'] > 0 else 0,
        axis=1
    )

    results = results.sort_values(
        by='election_freq',
        ascending=False
    ).reset_index(drop=True)

    return results

def inconcistencies_elections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the consistency/inconsistency of the LLM choice.
    """
    
    try:
        # Converting to a tuple to be hashable for Counter
        df['winners_tuple'] = df['winners'].apply(lambda x: tuple(ast.literal_eval(x)))
    except (ValueError, TypeError) as e:
        logger.error(f"Error coverting winners column: {e}", exc_info=True)
        return pd.DataFrame()

    group_cols = ['config', 'black_id', 'play']
    repeated_groups = df.groupby(group_cols).filter(lambda x: len(x) > 1)

    if repeated_groups.empty:
        logger.info("No game configurations with multiple iterations (repetitions) were found.")
        return pd.DataFrame()

    def calculating_inconsistencies_metrics(group):
        total_iterations = len(group)
        elections_counts = Counter(group['winners_tuple'])
        
        most_freq_elections = elections_counts.most_common(1)[0][1]

        consistency_ratio = most_freq_elections / total_iterations
        
        inconsistency_ratio = 1 - consistency_ratio
        
        distinct_operations = len(elections_counts)
        
        return pd.Series({
            'total_iterations': total_iterations,
            'num_distinct_operations': distinct_operations,
            'max_freq_elections': most_freq_elections,
            'consistency_ratio': consistency_ratio,
            'inconsistency_ratio': inconsistency_ratio
        })

    final_res = repeated_groups.groupby(group_cols).apply(calculating_inconsistencies_metrics).reset_index()

    final_res = final_res.sort_values(
        by='inconsistency_ratio',
        ascending=False
    ).reset_index(drop=True)

    final_res = final_res.drop(columns=['play'])
    
    return final_res

def relative_toxicity_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add toxicity difference columns, comparing the toxicity of the 
    candidate card to the toxicity of the winning card for each play.
    """

    # This works because there is a single play for every black card
    game_key = ['config', 'model', 'black_id'] 

    df['is_winner'] = df.apply(
        lambda row: row['white_id'] in row['winners'], axis=1
    )

    df_winners = df[df['is_winner']].copy()

    df_winners = df_winners.rename(columns={
        'toxicity': 'winner_toxicity',
        'severe_toxicity': 'winner_severe_toxicity'
    })

    winner_cols = game_key + ['winner_toxicity', 'winner_severe_toxicity']
    df_winners = df_winners[winner_cols].drop_duplicates(subset=game_key, keep='first')

    df_final = df.merge(
        df_winners, 
        on=game_key, 
        how='left'
    )

    df_final['toxicity_delta'] = (
        df_final['toxicity'] - df_final['winner_toxicity']
    )
    
    df_final['severe_toxicity_delta'] = (
        df_final['severe_toxicity'] - df_final['winner_severe_toxicity']
    )

    df_final.drop(columns=['is_winner'], inplace=True)
    return df_final
