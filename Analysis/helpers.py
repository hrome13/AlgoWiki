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
        df = pd.concat([df, copy.to_frame().T], ignore_index=True)
    return df

def clean_data():
    """
    Go through data.csv and if any algos have multiple variations, rewrite data.csv with the algo split into multiple rows.
    Get rid of trailing ?'s in the complexity class columns.
    """
    dataframe = pd.read_csv('Analysis/data.csv')
    dataframe = dataframe.replace(np.nan, '', regex=True)
    dataframe = dataframe[dataframe['Time Complexity Class'] != '']
    searching = True
    found = False
    while searching:
        for index, row in dataframe.iterrows():
            row['Space Complexity Class'] = str(row['Space Complexity Class']).replace('?', '')
            row['Time Complexity Class'] = str(row['Time Complexity Class']).replace('?', '')
            if '; ' in str(row['Variation']):
                dataframe = add_rows_for_algo(dataframe, row, index)
                found = True
                break
        if not found:
            searching = False
        else:
            found = False
    dataframe.to_csv('Analysis/data.csv')
    return dataframe

def get_variations(family):
    """
    For a given problem family, returns a list of all the variations of it that appear on the spreadsheet
    """
    dataframe = pd.read_csv('Analysis/data.csv')
    dataframe = dataframe.replace(np.nan, '', regex=True)
    algorithms = dataframe.loc[dataframe['Family Name'] == family]
    return algorithms['Variation'].unique()

def get_families():
    """
    Return a list of all the problem families that appear on the spreadsheet
    """
    dataframe = pd.read_csv('Analysis/data.csv')
    dataframe = dataframe.replace(np.nan, '', regex=True)
    return dataframe['Family Name'].unique()

# families = get_families()
# for fam in families:
#     print(fam + ": ")
#     print(get_variations(fam).tolist())