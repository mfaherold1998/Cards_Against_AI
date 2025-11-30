import ollama
from itertools import product
from typing import Dict, Literal
import pandas as pd

from src.utils.logging import create_logger
from src.utils.prompts import PROMPTS
logger = create_logger (log_name="main")

def single_round(model, prompt, temperature):

    """Run a single Ollama chat round."""
        
    model_response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature}  #"format": "json"
    )

    return model_response

def run_models(
        n_rounds: int,
        models: list[str],
        temperatures: list[float],
        games: Dict[str, pd.DataFrame],
        cards: dict,
        run_langs: str,
        prompt_to_use: Literal["prompt_player", "prompt_judge"],
        character_description: str
    ) -> pd.DataFrame:

    """
    Runs models on all game configurations, printing progress per console.
    """
    results = []

    for l in run_langs:
        
        total_combos = len(games[l]) * len(models) * len(temperatures)
        combo_count = 0

        logger.info(f"START: Running {total_combos} combinations for languague {l}. Each combination runs {n_rounds} rounds.")

        for (config_name, df_cfg), model, temperature in product(games[l].items(), models, temperatures):

            combo_count += 1
            logger.info(f"Processing: ({combo_count}/{total_combos})")
            logger.info(f"Config: {config_name} | Model: {model} | Temp: {temperature}")


            for play in df_cfg.values:                
                lang = play[0]
                assert lang==l
                black_card_id = play[1]
                white_card_ids = play[2:]               
                card_format = "ID:{id} TEXT:{text};"
                
                try:
                    
                    white_cards_options = [
                        card_format.format(id=card_id, text=cards[lang]["WHITE"].loc[card_id, 'card_text'])
                        for card_id in white_card_ids
                    ]
                    options_str = "".join(white_cards_options)
                    black_card_text = cards[lang]["BLACK"].loc[black_card_id, 'card_text']

                except KeyError as e:
                    logger.error(f"Missing card ID {e} in cards dict for lang {lang.upper()} (Play skipped).")
                    continue

                for i in range(n_rounds):
                    prompt =  PROMPTS[lang][prompt_to_use]                  
                    prompt = prompt.format(character_description=character_description,black_card_text=black_card_text, white_cards_options=options_str)

                    try:
                        res = single_round(
                            model, 
                            prompt, 
                            temperature
                        )
                        content = getattr(getattr(res, "message", None), "content", "")
                        
                    except Exception as e:
                        content = f"API_ERROR: {type(e).__name__}: {e}"                    
                        logger.error(f"Error during round {i+1} for {config_name}|{model}. {content}", exc_info=True)
                        
                    # 4. Acumular resultados
                    results.append({
                        "config": config_name,
                        "iteration": i + 1,
                        "lang": lang,
                        "model": model,                
                        "temperature": temperature,
                        "black_id": black_card_id,
                        "play": list(white_card_ids),
                        "response": content                           
                    })


    logger.info(f"All combinations completed. Total results: {len(results)} rows.")
    return pd.DataFrame(results)

