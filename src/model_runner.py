import ollama
from itertools import product
from typing import Dict
import pandas as pd

def single_round(model, prompt, temperature):

    """Run a single Ollama chat round."""
        
    model_response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature}
    )

    return model_response

def run_models(
        n_rounds: int,
        models: list[str],
        temperatures: list[float],
        pick: bool,
        games: Dict[str, pd.DataFrame],
        cards: dict
    ) -> pd.DataFrame:

    """
    Runs models on all game configurations, printing progress per console.
    """

    prompt = '''
                Hi! Let\'s play a round of Cards Against Humanity.
                I will be the Card Zar and you the player.
                The Black card I choose is: {black_card_text}.
                The White cards available are: {white_cards_options}.
                Which card do you choose? Remember that the goal is to choose the funniest one.
                If the black card have more than one space to fill, select a card for each space.
                As a response, give me just the id of the card (or cards) you choose in a format ID: id1, id2...and so on.  '''

    results = []

    total_combos = len(games) * len(models) * len(temperatures)
    combo_count = 0

    print(f"\n[START] Running {total_combos} configuration/model/temperature combinations...")
    print(f"[INFO] Each combination runs {n_rounds} rounds.\n")

    # games: Dict[str, pd.DataFrame]
    #   key = name of configuration (e.g. "RANDOM_5_EN")
    #   value = DataFrame of rows [lang, black_id, white_id_1, white_id_2, ...]

    for (config_name, df_cfg), model, temperature in product(games.items(), models, temperatures):

        combo_count += 1
        print(f"--- Processing ({combo_count}/{total_combos}) ---")
        print(f"Config: {config_name} | Model: {model} | Temp: {temperature}")
        print(f"Number of plays: {len(df_cfg)}\n")
        
        for play_idx, play in enumerate(df_cfg.values, start=1): 

            if not pick and play[1] in ["B005", "B091"]:
                print(f"[SKIP] {play[1]} requires >2 picks.")
                continue
            
            # Get cards for the prompt
            card_format = "ID:{id} TEXT:{text};"
            lang = play[0]  
            white_cards_options = [card_format.format(id=card_id, text=cards["W_"+lang][card_id]) for card_id in play[2:]]
            options_str = "".join(white_cards_options)
            
            for i in range(n_rounds):
                round_num = i + 1
                print(f"[ROUND {round_num}/{n_rounds}] Play {play_idx}/{len(df_cfg)} | Model: {model} | Config: {config_name}")
                try:
                    res = single_round( model, 
                                        prompt.format(black_card_text=cards["B_"+lang][play[1]], white_cards_options=options_str), 
                                        temperature)
                    content = getattr(getattr(res, "message", None), "content", "")
                except Exception as e:
                    content = f"ERROR: {type(e).__name__}: {e}"
                    print(f"[ERROR] during round {round_num}: {content}")

                results.append({
                                "config_name": config_name,
                                "iteration": i+1,
                                "language": lang,
                                "model": model,                
                                "temperature": temperature,    
                                "play": [card_id for card_id in play[1:]],
                                "response": content                           
                            })
                    
    
    print(f"[END] All combinations completed. Total results: {len(results)} rows.\n")
    return pd.DataFrame(results)