import ast
import pandas as pd

def parse_list_column(text):
    if pd.isna(text):
        return []
    try:
        parsed = ast.literal_eval(text)
        
        if isinstance(parsed, list):
            return parsed
        else:
            return [parsed]
    except (ValueError, SyntaxError):
        return []

def parse_ingredients(ingredients_str: str):
    return parse_list_column(ingredients_str)

def parse_steps(steps_str: str):
    return parse_list_column(steps_str)