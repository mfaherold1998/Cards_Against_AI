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
                As a response, give me just the id of the card (or cards) you choose in a format: "ID: id1, ID:id2..." and so on.  '''

    results = []
    total_combos = len(games) * len(models) * len(temperatures)
    combo_count = 0

    print(f"\n[START] Running {total_combos} combinations.")
    print(f"[INFO] Each combination runs {n_rounds} rounds.\n")

    
    for (config_name, df_cfg), model, temperature in product(games.items(), models, temperatures):

        combo_count += 1
        print(f"--- Processing ({combo_count}/{total_combos}) ---")
        print(f"Config: {config_name} | Model: {model} | Temp: {temperature}")

        for play in df_cfg.values:                
            lang = play[0]  
            black_card_id = play[1]
            white_card_ids = play[2:]               
            card_format = "ID:{id} TEXT:{text};"
            
            try:
                white_cards_options = [
                    card_format.format(id=card_id, text=cards["W_"+lang.upper()][card_id]) 
                    for card_id in white_card_ids
                ]
                options_str = "".join(white_cards_options)
                black_card_text = cards["B_"+lang.upper()][black_card_id]

            except KeyError as e:
                print(f"\n[ERROR] Missing card ID {e} in cards dict for lang {lang.upper()}. Skipping play.")
                continue

            for i in range(n_rounds):
                
                try:
                    res = single_round(
                        model, 
                        prompt.format(black_card_text=black_card_text, white_cards_options=options_str), 
                        temperature
                    )
                    content = getattr(getattr(res, "message", None), "content", "")
                    
                except Exception as e:
                    content = f"ERROR: {type(e).__name__}: {e}"
                    # Print errors, but do not interrupt the progress bar.
                    print(f"\n[ERROR] during round {i+1} for {config_name}|{model}: {content}")
                    
                # 4. Acumular resultados
                results.append({
                    "config": config_name,
                    "iteration": i + 1,
                    "lang": lang,
                    "model": model,                
                    "temperature": temperature,    
                    "play": [black_card_id] + list(white_card_ids),
                    "response": content                           
                })


    print(f"\n[END] All combinations completed. Total results: {len(results)} rows.")
    return pd.DataFrame(results)

