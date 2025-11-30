from pydantic import BaseModel, ConfigDict, model_validator
from typing import List, Literal

# Esquema para validar cada fila del Excel de cartas
class CardSchema(BaseModel):
    card_id: str
    card_text: str

# Esquema para validar cada fila del Excel de games
class GameSchema(BaseModel):
    # 1. Campos fijos que SIEMPRE deben estar y validarse
    lang: str
    black_id: str
    model_config = ConfigDict(extra='allow')

    @model_validator(mode='after')
    def check_dynamic_columns(self):
        # En Pydantic V2, los campos extra se guardan en self.model_extra
        # Si no hay extras, model_extra puede ser None, así que usamos "or {}"
        extra_data = self.model_extra or {}
        
        # Buscamos cuántas claves empiezan por 'white_id_'
        white_cols = [
            key for key, value in extra_data.items() 
            if key.startswith('white_id_') and value is not None
        ]

        # Validamos la regla de "al menos 2"
        if len(white_cols) < 2:
            raise ValueError(
                f"White card columns are missing."
                f"Two are required but there are: {len(white_cols)}: {white_cols}"
            )
        
        return self

# Esquema para validar las raw responses de los modelos
class RawResponsesSchema(BaseModel):
    config: str
    iteration: int
    lang: str
    model: str
    temperature: float
    black_id: str
    play: str
    response: str

# Esquema para validar WinnersSentences
class WinnersSentencesSchema(BaseModel):
    config: str
    iteration: int
    lang: str
    model: str
    temperature: float
    black_id: str
    play: str
    winners: str
    sentence: str

# Esquema para validar AllCombinationSentences
class AllCombinationSchema(BaseModel):
    config: str
    lang: str
    model: str
    temperature: float
    winners: str
    play: str
    black_id: str
    white_id: str
    sentence: str

# Esquema para validar DetoxifyAllCombSchema
class DetoxifyAllCombSchema(AllCombinationSchema):
    toxicity: float
    severe_toxicity: float
    obscene: float
    threat: float
    insult: float
    identity_attack: float
    severe_toxicity_gt_05: int
    severe_toxicity_gt_08: int

# Esquema para validar DetoxifyWinners
class DetoxifyWinnersSchema(WinnersSentencesSchema):
    toxicity: float
    severe_toxicity: float
    obscene: float
    threat: float
    insult: float
    identity_attack: float
    severe_toxicity_gt_05: int
    severe_toxicity_gt_08: int

# Esquema para validar DetoxifyAllCombSchema
class PerspectiveAllCombSchema(AllCombinationSchema):
    toxicity: float
    severe_toxicity: float
    obscene: float
    threat: float
    insult: float
    identity_attack: float
    severe_toxicity_gt_05: int
    severe_toxicity_gt_08: int

# Esquema para validar DetoxifyWinners
class PerspectiveWinnersSchema(WinnersSentencesSchema):
    toxicity: float
    sexually_explicit: float
    severe_toxicity: float
    obscene: float
    threat: float
    profanity: float
    insult: float
    identity_attack: float
    severe_toxicity_gt_05: int
    severe_toxicity_gt_08: int

# Esquema para validar el JSON de configuración 1
class JsonConfigSchema_1(BaseModel):
    cards_dir: str
    results_dir: str
    file_type: Literal["xlsx", "csv"]
    languages: List[str]
    models: List[str]
    temperatures: List[float]
    rounds: int
    dataset_mode: Literal['test', 'all']
    test_num_rows: int
    prompt_type: Literal['prompt_player', 'prompt_judge']
    character_description: str

# Esquema para validar el JSON de configuración 2
class JsonConfigSchema_2(BaseModel):
    run_id: str
    run_config_dir: str

# Esquema para validar el JSON de configuración 3
class JsonConfigSchema_3(BaseModel):
    run_id: str
    run_config_dir: str
    detoxify_model: str
    device: str
    batch_size: int

# Esquema para validar el JSON de configuración 5
class JsonConfigSchema_5(BaseModel):
    analysis_dir: str
    file_type: str