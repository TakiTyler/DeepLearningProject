import os
import pandas as pd
from constants import * # paths to CSVs in here

# helper function to print out CSVs for debugging purposes
def debug_CSVs(csv: str, indexes: list = None):
    if not os.path.exists(csv): raise ValueError('Input a proper CSV file location.')

    df = pd.read_csv(csv)
    if indexes != None: print(df.loc[indexes])
    else: print(df)

# helper function to load CSVs for use in other functions
def load_csv(csv: str, indexes: list = None):
    if not os.path.exists(csv): raise ValueError('Input a proper CSV file location.')

    df = pd.read_csv(csv)
    if indexes != None: return df.loc[indexes]
    else: return df

if __name__ == '__main__':
    
    df = load_csv(RAW_RECI)
    print(df.columns)
    