import os
import pandas as pd
from constants import * # paths to CSVs in here
from preprocessing.parse import parse_ingredients, parse_list_column, parse_steps
from preprocessing.clean import clean_ingredients, clean_steps, is_valid_recipe

# helper function to load CSVs for use in other functions
def load_csv(csv: str):
    if not os.path.exists(csv): 
        raise ValueError('Input a proper CSV file location.')
    df = pd.read_csv(csv)
    return df

# test methods
def demo_parsing(df):
    print("=== PARSING DEMO ===")

    ingredients = parse_ingredients(df.iloc[0]['ingredients'])
    steps = parse_steps(df.iloc[0]['steps'])
    tags = parse_list_column(df.iloc[0]['tags'])

    print("INGREDIENTS:", ingredients)
    print("STEPS:", steps[:2])  # just preview
    print("TAGS:", tags)
    print()


def demo_cleaning(df):
    print("=== CLEANING DEMO ===")

    ingredients = parse_ingredients(df.iloc[0]['ingredients'])
    steps = parse_steps(df.iloc[0]['steps'])

    ingredients = clean_ingredients(ingredients)
    steps = clean_steps(steps)

    print("CLEAN INGREDIENTS:", ingredients)
    print("CLEAN STEPS:", steps[:2])
    print("VALID:", is_valid_recipe(ingredients, steps))
    print()


if __name__ == '__main__':
    df = load_csv(RAW_RECI)
    '''
    # demo parsing and cleaning on first recipe
    demo_parsing(df)
    demo_cleaning(df)
    '''
    