
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import helpers

def no_tradeoff_necessary(family_name, variation):
    """
    Returns list of algorithm names if there are algorithms (among the algorithms that we have analyzed the space)
        for a problem variation that are both the fastest and most space efficient.
    Otherwise, returns []

    Prints this list as well.
    """
    dataframe = pd.read_csv('data.csv')
    dataframe = dataframe.replace(np.nan, '', regex=True)
    algorithms = dataframe.loc[dataframe['Family Name'] == family_name]
    algorithms = algorithms.loc[algorithms['Variation'] == variation]
    
    if algorithms.empty:
        print('No data found for family name: ' +
              family_name + ' and variation: ' + variation)
        return False

    algorithms = algorithms[algorithms['Space Complexity Class'].str.contains(':')].sort_values('Year')

    best_time = 8
    best_time_algs = []
    best_space = 8
    best_space_algs = []
    algs = []

    for index, row in algorithms.iterrows():
        item = {}
        item['year'] = int(row['Year'])
        spaces = row['Space Complexity Class'].split(',\n')
        for dependency in spaces:
            dependency_list = dependency.split(': ')
            if dependency_list[0] == 'n' or dependency_list[0] == 'V':
                item['space'] = float(dependency_list[1]) - 1 # TODO: CEIL THIS (ROUND UP)
        item['space'] = item.get('space', 0)
        item['time'] = float(row['Time Complexity Class']) - 1 # TODO: CEIL THIS (ROUND UP)
        item['name'] = row['Algorithm Name']

        algs.append(item['name'])

        if item['space'] < best_space:
            best_space_algs = [item['name']]
            best_space = item['space']
        elif item['space'] == best_space:
            best_space_algs.append(item['name'])

        if item['time'] < best_time:
            best_time_algs = [item['name']]
            best_time = item['time']
        elif item['time'] == best_time:
            best_time_algs.append(item['name'])

    best = []
    for alg in algs:
        if alg in best_time_algs and alg in best_space_algs:
            best.append(alg)
    print(best)
    return best

def fraction_of_optimality():
    total = 0
    has_optimal = 0
    families = helpers.get_families()
    for family in families:
        variations = helpers.get_variations(family)
        for variation in variations:
            total += 1
            optimal_algs = no_tradeoff_necessary(family, variation)
            if optimal_algs:
                has_optimal += 1
    return (total, has_optimal, has_optimal/total)


# family = 'Maximum Subarray Problem'
# variation = '1D Maximum Subarray'
# family = 'Sorting'
# variation = 'Non-Comparison Sorting'
# no_tradeoff_necessary(family, variation)
print(fraction_of_optimality())