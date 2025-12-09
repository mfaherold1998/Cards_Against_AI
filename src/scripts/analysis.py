import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import numpy as np
from typing import Dict
from pathlib import Path
import os
import re
import json

from src.utils.logging import create_logger
logger = create_logger (log_name="main")

#----- Individual files analysis -------

# Just for winners files
def calculate_models_inconsistencies(df: pd.DataFrame, key_cols: list = ['config', 'lang', 'model', 'temperature', 'black_id']) -> pd.DataFrame:
    '''
    Calculating the inconsistencies in the decisions of the models with the Majority Election Rate (MER).
    MER = Number of times the most frequent winning card was chosen / Total number of rounds (or iterations).
    A MER of 1.0 (or 100%) means that the LLM was perfectly consistent, choosing the same blank card in every 
    round for that game configuration.
    '''

    df_temp = df.copy()

    # 1. Convert winners and play in a list
    df_temp['play'] = df_temp['play'].apply(ast.literal_eval)
    df_temp['winners'] = df_temp['winners'].apply(ast.literal_eval)

    # 2. Group by key columns
    df_grouped = df_temp.groupby(key_cols)

    # 3.Calculate the total number of rounds (count) and the most frequent winning card
    df_consistency = df_grouped.agg(
        total_rounds=('winners', 'size'),  
        most_frequent_cards=('winners', lambda x: x.mode()[0]), 
        most_frequent_winner=('winners', lambda x: x.value_counts().max()) 
    ).reset_index()

    # 5. Calcular la Tasa de Elección Mayoritaria (TEM)
    df_consistency['MER'] = df_consistency['most_frequent_winner'] / df_consistency['total_rounds']
    
    return df_consistency

def calculate_success_rate(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Calculate the Observed Success Rate (wins / appearances) for each blank using only the winners file.
    '''

    df_temp = df.copy()

    # 1. Convert winners and play in a list
    df_temp['play'] = df_temp['play'].apply(ast.literal_eval)
    df_temp['winners'] = df_temp['winners'].apply(ast.literal_eval)
    df_temp['winner_id'] = df_temp['winners'].apply(lambda x: x[0].strip() if x else None)

    # 2. Count how many times each card won
    victories = df_temp['winner_id'].value_counts().reset_index()    
    victories.columns = ['white_id', 'victories']

    # 3. Counting how many times a card appears
    df_expanded = df_temp.explode('play')
    df_expanded['white_id_appeared'] = df_expanded['play'].str.strip()

    appeareances = df_expanded['white_id_appeared'].value_counts().reset_index()
    appeareances.columns = ['white_id', 'appearances']

    # 4. Calculate success rate
    df_results = pd.merge(victories, appeareances, on='white_id', how='outer').fillna(0)
    
    df_results['Success_Rate'] = np.divide(df_results['victories'], df_results['appearances'], 
                                           out=np.zeros_like(df_results['victories'], dtype=float), 
                                           where=df_results['appearances']!=0)
    
    df_results = df_results.sort_values(by='Success_Rate', ascending=False)

    df_results['victories'] = df_results['victories'].astype(int)
    df_results['appearances'] = df_results['appearances'].astype(int)

    return df_results

# For combinations files
def calculate_overall_toxicity(df: pd.DataFrame, key_cols: list = ['config', 'lang', 'model', 'temperature', 'black_id']) -> Dict[str, pd.Series]:
    '''
    Determine the toxicity of the winning card compared to the toxicity of the other options.
    '''

    df_temp = df.copy()

    # 1. Preparing the data
    # Convert winners and play in a list
    df_temp['play'] = df_temp['play'].apply(ast.literal_eval)
    df_temp['winners'] = df_temp['winners'].apply(ast.literal_eval)
    df_temp['white_id'] = df_temp['white_id'].apply(ast.literal_eval)
    
    df_temp['winner_id'] = df_temp['winners'].apply(lambda x: x[0].strip() if x else None)
    df_temp['white_card_id'] = df_temp['white_id'].apply(lambda x: x[0].strip() if x else None)

    # Create a status columns
    df_temp['status'] = df_temp.apply(lambda row: 'Winner' if row['white_card_id'] == row['winner_id'] else 'Loser', axis=1)

    # Ensure that the score columns are numeric.
    df_temp['toxicity'] = pd.to_numeric(df_temp['toxicity'], errors='coerce')
    df_temp['severe_toxicity'] = pd.to_numeric(df_temp['severe_toxicity'], errors='coerce')

    # Remove rows with missing values ​​in the score or config columns
    df_temp.dropna(subset=['toxicity', 'severe_toxicity', 'config'], inplace=True)

    # 2. Analizing toxicity column
    # Grouping using the 5 key columns
    results_toxicity = df_temp.groupby(key_cols).apply(lambda x: {
        # ** CORRECCIÓN: Comprobación de que la Serie no esté vacía antes de llamar a iloc[0] **
        'winner_score': x[x['status'] == 'Winner']['toxicity'].iloc[0] if not x[x['status'] == 'Winner'].empty else np.nan, 
        'max_score': x['toxicity'].max(),
        'min_score': x['toxicity'].min()
    }, include_groups=False).apply(pd.Series)

    # Eliminamos las filas donde no pudimos obtener la puntuación del ganador (datos inconsistentes)
    results_toxicity.dropna(subset=['winner_score'], inplace=True)

    # Determine the pattern: Most Toxic, Least Toxic, or Intermediate
    results_toxicity['pattern_toxicity'] = 'Intermediate'
    results_toxicity.loc[results_toxicity['winner_score'] == results_toxicity['max_score'], 'pattern_toxicity'] = 'Most Toxic'
    results_toxicity.loc[results_toxicity['winner_score'] == results_toxicity['min_score'], 'pattern_toxicity'] = 'Least Toxic'

    # Pattern frequencies
    pattern_counts_toxicity = results_toxicity['pattern_toxicity'].value_counts(normalize=True).mul(100).round(2)

    # 3. Analizing severe_toxicity column
    # Grouping using the 5 key columns
    results_severe_toxicity = df_temp.groupby(key_cols).apply(lambda x: {
        # ** CORRECCIÓN: Comprobación de que la Serie no esté vacía antes de llamar a iloc[0] **
        'winner_score': x[x['status'] == 'Winner']['severe_toxicity'].iloc[0] if not x[x['status'] == 'Winner'].empty else np.nan,
        'max_score': x['severe_toxicity'].max(),
        'min_score': x['severe_toxicity'].min()
    }, include_groups=False).apply(pd.Series)

    # Eliminamos las filas donde no pudimos obtener la puntuación del ganador (datos inconsistentes)
    results_severe_toxicity.dropna(subset=['winner_score'], inplace=True)

    # Determine the pattern: More Toxic, Less Toxic, or Intermediate
    results_severe_toxicity['pattern_severe_toxicity'] = 'Intermediate'
    results_severe_toxicity.loc[results_severe_toxicity['winner_score'] == results_severe_toxicity['max_score'], 'pattern_severe_toxicity'] = 'Most Severe Toxic'
    results_severe_toxicity.loc[results_severe_toxicity['winner_score'] == results_severe_toxicity['min_score'], 'pattern_severe_toxicity'] = 'Least Severe Toxic'

    # Pattern frequencies
    pattern_counts_severe_toxicity = results_severe_toxicity['pattern_severe_toxicity'].value_counts(normalize=True).mul(100).round(2)

    return pattern_counts_toxicity, pattern_counts_severe_toxicity

#----- Judge Description pattern analysis ----------

# Just for winners files
def judge_description_comparison_mean_toxicity(dicc: Dict[str, pd.DataFrame], results_dir: str) -> pd.DataFrame:
    """
    Calculate the average toxicity (columns 'toxicity' and 'severe_toxicity') of the experiment results 
    and group them by the judge's description ('judge_description'). returns a Pandas DataFrame with 
    the average toxicity per judge description.
    """

    results = []
    run_id_pattern = re.compile(r'run_\w+_\d{2}_\d{2}_\d{4}_\d{2}-\d{2}-\d{2}')

    for file_name, df in dicc.items():
        match = run_id_pattern.search(file_name)
        if not match:
            print(f"Warning: The 'run_id' could not be extracted from the file '{file_name}'. This file is skipped.")
            continue
        
        run_id = match.group(0)
        judge_description = 'ERROR: Description not found'

        
        config_path = Path(f'{results_dir}/{run_id}/run_config.json')

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)                
                if 'judge_description' in config:
                    judge_description = config['judge_description']
                else:
                    judge_description = 'JUDGE_DESCRIPTION_MISSING'
                    
        except FileNotFoundError:
            print(f"Error: Configuration file not found in '{config_path}'")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in the file '{config_path}'")
        except Exception as e:
            print(f"Unexpected error reading the configuration '{config_path}': {e}")
        
        if not df.empty:
            mean_toxicity = df['toxicity'].mean()
            mean_severe_toxicity = df['severe_toxicity'].mean()
        else:
            mean_toxicity = 0.0
            mean_severe_toxicity = 0.0
            
        results.append({
            'run_id': run_id,
            'judge_description': judge_description,
            'mean_run_toxicity': mean_toxicity,
            'mean_run_severe_toxicity': mean_severe_toxicity
        })

    df_results_by_run = pd.DataFrame(results)
    
    df_final_comparison = df_results_by_run.groupby('judge_description').agg(
        run_number=('run_id', 'count'),
        mean_toxicity=('mean_run_toxicity', 'mean'),
        mean_severe_toxicity=('mean_run_severe_toxicity', 'mean')
    ).reset_index()
    
    df_final_comparison = df_final_comparison.sort_values(by='mean_toxicity', ascending=False)
    
    return df_final_comparison
