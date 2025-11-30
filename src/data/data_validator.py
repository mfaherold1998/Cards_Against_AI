import pandas as pd
import numpy as np
from pydantic import ValidationError
from typing import List, Dict, Type, Tuple, Any
from enum import Enum

from src.utils.logging import create_logger
logger = create_logger (log_name="main")

# --- VALIDATION LOGIC ---

def clean_dataframe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Take a pandas dataframe, remove all occurrences of NAN, and convert the df to a dictionary. 
    Pydantic handles 'None' better than 'NaN' (float).
    """
    # Reemplaza todos los NaN por None
    df_clean = df.replace({np.nan: None})
    
    # Convierte a lista de dicts: [{'col1': val1}, {'col1': val2}...]
    return df_clean.to_dict(orient='records')

def validate_dataframe(df: pd.DataFrame, schema_model: Type[Any]) -> Tuple[List[Any], List[str]]:
    """
    Generic Function: Accepts any DataFrame and any Pydantic model.
    Returns:
    - A list of validated objects (model instances).
    - A list of errors (strings describing row and error).
    """

    records = clean_dataframe(df)
    valid_objects = []
    errors = []

    for index, record in enumerate(records):
        try:
            # Funciona para columnas fijas y dinámicas (extra='allow').
            obj = schema_model(**record)
            valid_objects.append(obj)
            
        except ValidationError as e:
            # Formateamos el error para que sea legible
            error_msgs = "; ".join([f"{err['loc'][0]}: {err['msg']}" for err in e.errors()])
            human_error = f"Fila {index + 2} (Excel): {error_msgs}" # +2 porque Pandas empieza en 0 y Excel tiene header
            errors.append(human_error)
        
        except ValueError as e:
            # Para errores manuales lanzados en validadores custom (como el de white_ids)
            errors.append(f"Fila {index + 2} (Excel): {str(e)}")

    return valid_objects, errors

# --- HEAD DISPATCHER ---

def file_validator(df: pd.DataFrame, schema: Type[Any]) -> Dict[str, Any]:
    """
    Main function that acts as the entry point.
    
    Args:
        df: The loaded Pandas DataFrame.
        schema: schema for validate the df
        
    Return: 
    {
        "status": "success" | "partial_error" | "fatal_error"
        "message": "Successful validation" | "errors"
        "data": valid_data,
        "errors": validation_errors
    }
    """
    
    selected_schema = schema

    # 3. Llamar a la validación genérica
    logger.info(f"Validating file using schema: '{selected_schema.__name__}'...")
    valid_pydantic_objects, validation_errors = validate_dataframe(df, selected_schema) # Nombre cambiado para claridad

    # **CAMBIO CLAVE:** Convertir la lista de objetos Pydantic a una lista de diccionarios
    # y luego a un DataFrame. Usamos .model_dump() para Pydantic v2 o .dict() para v1.
    if hasattr(valid_pydantic_objects[0], 'model_dump'):
        # Pydantic v2
        valid_data_dicts = [obj.model_dump() for obj in valid_pydantic_objects]
    elif valid_pydantic_objects and hasattr(valid_pydantic_objects[0], 'dict'):
        # Pydantic v1
        valid_data_dicts = [obj.dict() for obj in valid_pydantic_objects]
    else:
        # En caso de lista vacía o modelos inesperados
        valid_data_dicts = []

    reconstructed_df = pd.DataFrame(valid_data_dicts)

    # 4. Construir respuesta
    if validation_errors:
        mes = f"There were found {len(validation_errors)} validation errors."
        logger.error(mes)
        return {
            "status": "partial_error" if not reconstructed_df.empty else "fatal_error",
            "message": mes,
            "data": reconstructed_df, # Devolvemos el DF reconstruido
            "errors": validation_errors
        }
    
    mes = "Successful validation"
    logger.info(mes)
    return {
        "status": "success",
        "message": mes,
        "data": reconstructed_df, # Devolvemos el DF reconstruido
        "errors": []
    }