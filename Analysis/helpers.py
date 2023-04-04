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
    dataframe = pd.read_csv('Analysis/data_dirty.csv')
    dataframe = dataframe.replace(np.nan, '', regex=True)
    dataframe = dataframe[(dataframe['Time Complexity Class'] != '#VALUE!') &
                          (dataframe['Space Complexity Class'] != '#VALUE!') &
                          (dataframe["Looked at?"] != 0.001) &
                          (dataframe["Exact Problem Statement?"] == 1)]

    # Get rid of parallel, quantum, and approximate algorithms
    dataframe = dataframe[(dataframe["Quantum?"] == 0) | (dataframe["Quantum?"] == "0")]
    dataframe = dataframe[(dataframe["Parallel?"] == 0) | (dataframe["Parallel?"] == "0")]
    dataframe = dataframe[(dataframe["Approximate?"] == 0) | (dataframe["Approximate?"] == "0")]

    searching = True
    found = False
    while searching:
        for index, row in dataframe.iterrows():
            # if row["Looked at?"] == "0.001" or row["Exact Problem Statement?"] == "0":
            #     continue
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

def get_families(data=None):
    """
    Return a list of all the problem families that appear on the spreadsheet
    """
    if data is None:
        dataframe = pd.read_csv('Analysis/data.csv')
    else:
        dataframe = data
    dataframe = dataframe.replace(np.nan, '', regex=True)
    return dataframe['Family Name'].unique()

def get_domains(data=None):
    """
    Return a list of all the problem domains that appear on the spreadsheet
    """
    if data is None:
        dataframe = pd.read_csv('Analysis/data.csv')
    else:
        dataframe = data
    dataframe = dataframe.replace(np.nan, '', regex=True)
    return dataframe['Domains'].unique()

# families = get_families()
# for fam in families:
#     print(fam + ": ")
#     print(get_variations(fam).tolist())
clean_data()

print(get_domains())
# df = pd.read_csv("Analysis/data_dirty.csv")
# cleaned = pd.read_csv("Analysis/data.csv")
# print(df["Family Name"].unique() - cleaned["Family Name"].unique())