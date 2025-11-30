import pandas as pd
import os
from pathlib import Path
from typing import List, Tuple
from src.data.data_validator import file_validator
from src.utils.utils import DataType
from src.data.schema import *

from src.utils.logging import create_logger
logger = create_logger (log_name="main")

# 1. Configurar el mapa de esquemas
SCHEMA_MAP = {
    DataType.CARDS.value: CardSchema,          
    DataType.GAMES.value: GameSchema,
    DataType.RAW_RESPONSES.value: RawResponsesSchema,
    DataType.COMBINATION_SENTENCES.value: AllCombinationSchema,
    DataType.WINNER_SENTENCES.value: WinnersSentencesSchema,
    DataType.COMBINATIONS_DETOX.value: DetoxifyAllCombSchema,
    DataType.COMBINATIONS_PERS.value: PerspectiveAllCombSchema,
    DataType.WINNERS_DETOX.value: DetoxifyWinnersSchema,
    DataType.WINNERS_PERS.value: PerspectiveWinnersSchema
}

def read_safe(file_path: Path) -> pd.DataFrame:
    ''' 
    Determine the right read funtion to use: read_excel or read_csv.
    Return => pd:DataFrame
    '''

    # 2. Controlar la existencia del fichero
    if not file_path.is_file():
        logger.error(f"File not found in: {file_path}")
        raise FileNotFoundError(f"File not found in: {file_path}")

    # 3. Determinar el tipo de archivo y seleccionar la función de lectura
    extension = file_path.suffix.lower()
    
    if extension == '.csv':
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"Error reading {file_path}. Details: {e}")
            raise IOError(f"Error reading {file_path}.")
            
    elif extension == '.xlsx':
        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            logger.error(f"Error reading {file_path}. Details: {e}")
            raise IOError(f"Error reading {file_path}")
            
    else:
        logger.error(f"Error reading {file_path}, not supported extension: '{extension}'. Accepted only .csv and .xlsx.")
        raise ValueError(f"Error reading {file_path}, not supported extension.")
        
    # 4. Retornar el DataFrame leído
    logger.info(f"File {file_path.name} read successfully. Rows: {len(df.index)}, Columns: {len(df.columns)}")
    return df

def determine_data_schema(file_path: Path) -> BaseModel:
    '''
    Determine the type of dataset: cards, games, configurations, etc...
    '''

    fname = file_path.stem.lower() # 'black_cards'
    dataset_types = SCHEMA_MAP.keys()

    for type in dataset_types:
        if type in fname:
            return SCHEMA_MAP[type]
        
    logger.error(f"The dataset type for the file could not be determined: {file_path}. Type is not in the list.")
    raise ValueError(f"Dataset type not found: {file_path}")

def load_data(file_path: str|Path) -> Tuple[pd.DataFrame, List[str]]:
    ''' Receive any file path and:
        - determine file type for correct data loading: ['xlsx', 'csv'],
        - determine dataset type for correct schema validation: ['cards', 'games', 'config']
    '''
    # 1. Convertir el str en Path
    path = Path(file_path)

    # 2. Determinar el esquema que corresponde al tipo de dato
    schema = determine_data_schema(path)

    # 3. Leer los datos en modo seguro
    df_data = read_safe(path)

    # 4. Validar que las columnas del df son correctas segun su esquema
    results_dict = file_validator(df_data, schema)

    # 5. Retornar los resultados
    return results_dict["data"], results_dict["errors"]
    
    

        

