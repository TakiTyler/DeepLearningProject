import os
import pandas as pd
from constants import * # paths to CSVs in here
from preprocessing.parse import parse_ingredients, parse_list_column, parse_steps
 
# helper function to load CSVs for use in other functions
def load_csv(csv: str):
    if not os.path.exists(csv): 
        raise ValueError('Input a proper CSV file location.')
    df = pd.read_csv(csv)
    return df

if __name__ == '__main__':
    df = load_csv(RAW_RECI)
    '''
    # example of how to use the parsing functions to get ingredients, steps, and tags from the RAW_RECI CSV
    # feel free to run it to understand it
    ingredients = parse_ingredients(df.iloc[0]['ingredients'])
    steps = parse_steps(df.iloc[0]['steps'])
    tags = parse_list_column(df.iloc[0]['tags'])

    print(ingredients)
    print(steps)
    print(tags)
    '''