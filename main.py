import os
import pandas as pd
from constants import * # paths to CSVs in here

def debug_CSVs(csv: str, indexes: list = None):
    if not os.path.exists(csv): raise ValueError('Input a proper CSV file location.')

    df = pd.read_csv(csv)
    if indexes != None: print(df.loc[indexes])
    else: print(df)

if __name__ == '__main__':

    debug_CSVs(INTER_VAL)