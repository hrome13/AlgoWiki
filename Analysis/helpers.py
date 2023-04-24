import pandas as pd
import numpy as np
import math

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

def get_algorithms_for_family(family_name, data=None):
    if data is None:
        data = pd.read_csv('Analysis/data.csv')
        data = data.replace(np.nan, '', regex=True)

    algorithms = data.loc[data['Family Name'] == family_name]
    return algorithms

def get_best_space(algorithms):
    """Get the best space complexity from the input algorithms (e.g. from a family or variation)"""
    best_space = 8
    for index, row in algorithms.iterrows():
        item = {}
        item['year'] = int(row['Year'])
        item['space'] = math.ceil(float(row['Space Complexity Class']) - 1)
        item['name'] = row['Algorithm Name']

        if item['space'] < best_space:
            best_space = item['space']

    return best_space

def get_best_time(algorithms):
    """Get the best time complexity from the input algorithms (e.g. from a family or variation)"""
    best_time = 8
    for index, row in algorithms.iterrows():
        item = {}
        item['year'] = int(row['Year'])
        item['time'] = math.ceil(float(row['Time Complexity Class']) - 1)
        item['name'] = row['Algorithm Name']

        if item['time'] < best_time:
            best_time = item['time']

    return best_time


def get_best_space_and_time(algorithms):
    best_space = get_best_space(algorithms)
    best_time = get_best_time(algorithms)
    return best_space, best_time

def get_all_best_space_and_time():
    data = pd.read_csv('Analysis/data.csv')
    data = data.replace(np.nan, '', regex=True)
    count = 0
    skipped = 0
    for fam in get_families():
        algorithms = get_algorithms_for_family(fam, data)
        if algorithms.empty:
            skipped += 1
            continue
        best_space = get_best_space(algorithms)
        best_time = get_best_time(algorithms)
        if best_space > best_time:
            # print(fam)
            continue
        count += 1
        print(f"{fam}\t{best_space}\t{best_time}")
    print(f"Count: {count}")
    print(f"Skipped: {skipped}")

clean_data()

# def get_num_steps(n, complexity_classification):
#     if n < 1:

# families = get_families()
# for fam in families:
#     algorithms = 
#     print(f"{fam}/")
#     print(get_variations(fam).tolist())

# get_all_best_space_and_time()

# print(get_domains())
# df = pd.read_csv("Analysis/data_dirty.csv")
# cleaned = pd.read_csv("Analysis/data.csv")
# print(df["Family Name"].unique() - cleaned["Family Name"].unique())