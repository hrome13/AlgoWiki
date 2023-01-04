import pandas as pd
import numpy as np

def add_rows_for_algo(df, row, idx):
    """
    If an algo has multiple variations (delimiter is '; '), split its row into multiple rows
    """
    df = df.drop([idx])
    vars = row['Variation'].split('; ')
    for var in vars:
        copy = row.copy(deep=True)
        copy['Variation'] = var
        # print(copy['Variation'])
        df = pd.concat([df, copy.to_frame().T], ignore_index=True)
    return df

def clean_data():
    """
    Go through data.csv and if any algos have multiple variations, rewrite data.csv with the algo split into multiple rows
    """
    dataframe = pd.read_csv('data.csv')
    dataframe = dataframe.replace(np.nan, '', regex=True)
    searching = True
    found = False
    while searching:
        for index, row in dataframe.iterrows():
            if '; ' in str(row['Variation']):
                dataframe = add_rows_for_algo(dataframe, row, index)
                found = True
                break
        if not found:
            searching = False
        else:
            found = False
    dataframe.to_csv('data.csv')
    return dataframe

def get_variations(family):
    """
    For a given problem family, returns a list of all the variations of it that appear on the spreadsheet
    """
    dataframe = pd.read_csv('data.csv')
    dataframe = dataframe.replace(np.nan, '', regex=True)
    algorithms = dataframe.loc[dataframe['Family Name'] == family]
    return algorithms['Variation'].unique()

def get_families():
    """
    Return a list of all the problem families that appear on the spreadsheet
    """
    dataframe = pd.read_csv('data.csv')
    dataframe = dataframe.replace(np.nan, '', regex=True)
    return dataframe['Family Name'].unique()

# families = get_families()
# for fam in families:
#     print(fam + ": ")
#     print(get_variations(fam).tolist())