from enum import Enum
from dotenv import load_dotenv, find_dotenv


class FilesNames(Enum):
    """Standard filenames for datasets."""
    BLACK_CARDS = "black_cards"
    WHITE_CARDS = "white_cards"

class DirNames(Enum):
    """Standard directory names for datasets."""
    CARDS_DIR = "cards_texts"
    GAMES_DIR = "games_config"
    TO_JUDGE_DIR = "to_judge_config"
    LLM_RAW_RESPONSES = "raw_responses"
    LLL_PROCESSED_DATA = "processed_data"
    LLL_TOXICITY_SCORES = "toxicity_scores"
    PLOTS = "plots"
    ANALYSIS = "analysis_module"
    ANALYSIS_RES = "analysis_results"

class ResultsNames(Enum):
    """Standard names for modules results."""
    LLM_RAW_RESPONSES = "all_models_raw_responses"
    NO_ID_RESPONSES = "all_games_no_id_detected"
    MISMATCH_RESPONSES = "all_games_mismatch"
    GOOD_RESPONSES = "winners_sentences"
    ALL_POSIBLE_COMBINATIONS = "all_combination_sentences"
    DETOXIFY_SCORES_WINNERS = "winners_detoxify_scores"
    DETOXIFY_SCORES_COMBINATIONS = "combinations_detoxify_scores"
    PERSPECTIVE_SCORES_WINNERS = "winners_perspective_scores"
    PERSPECTIVE_SCORES_COMBINATIONS = "combinations_perspective_scores"

class DataType(Enum):
    CARDS = "cards"
    GAMES = "games"
    CONFIG = "config"
    RAW_RESPONSES = "raw_responses"
    COMBINATION_SENTENCES = "combination_sentences"
    WINNER_SENTENCES = "winners_sentences"
    COMBINATIONS_DETOX = "combinations_detoxify"
    COMBINATIONS_PERS = "combinations_perspective"
    WINNERS_DETOX = "winners_detoxify"
    WINNERS_PERS = "winners_perspective"

class ToxicityAttributes(Enum):
    """Standard names for toxicity attributes to measure."""
    TOXICITY = "toxicity"
    SEVERE_TOXICITY = "severe_toxicity"
    OBSCENE = "obscene"
    THREAT = "threat" 
    INSULT = "insult"
    IDENTITY_ATTACK = "identity_attack"
    SEXUALLY_EXPLICIT = "sexually_explicit"
    PROFANITY = "profanity"

class ClasifiersNames(Enum):
    DETOXIFY = "detoxify"
    PERSPECTIVE = "perspective"

def load_env() -> None:
    
    found = find_dotenv(usecwd=True)
    if found:
        load_dotenv(found)
        return
    
def build_play_key (black_id: str, winners: list) -> str: # B1|W1,W2,W3...
    return f"{black_id}|{winners}"