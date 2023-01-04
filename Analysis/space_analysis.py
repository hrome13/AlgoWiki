import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import helpers
import seaborn as sns
from tabulate import tabulate

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
                item['space'] = math.ceil(float(dependency_list[1]) - 1) # TODO: CEIL THIS (ROUND UP)
        item['space'] = item.get('space', 0)
        try:
            item['time'] = float(row['Time Complexity Class']) - 1 #math.ceil(float(row['Time Complexity Class']) - 1)
        except:
            # print(family_name, variation, row['Algorithm Name'], row['Time Complexity Class'])
            return
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

    return best

def fraction_of_optimality():
    """
    Determine the fraction of variants that have an algorithm that is both
    faster and more space efficient (asymptotically) than the other algorithms
    """
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

def space_improvements_by_year(family_name, variation):
    dataframe = pd.read_csv('data.csv')
    dataframe = dataframe.replace(np.nan, '', regex=True)
    algorithms = dataframe.loc[dataframe['Family Name'] == family_name]
    if variation != "by family":
        algorithms = algorithms.loc[algorithms['Variation'] == variation]
    
    if algorithms.empty:
        print('No data found for family name: ' +
              family_name + ' and variation: ' + variation)
        return False

    algorithms = algorithms[algorithms['Space Complexity Class'].str.contains(':')].sort_values('Year')

    # Get the algorithms that improve the space bounds
    improvements = []
    best_space = None
    algorithms = algorithms.sort_values('Year')
    for index, row in algorithms.iterrows():
        item = {}
        item['year'] = int(row['Year'])
        spaces = row['Space Complexity Class'].split(',\n')

        # Get the space complexity class that is relavent
        for dependency in spaces:
            dependency_list = dependency.split(': ')
            if dependency_list[0] == 'n' or dependency_list[0] == 'V':
                item['space'] = float(dependency_list[1]) - 1 # math.ceil(float(dependency_list[1]) - 1)
        item['space'] = item.get('space', 0)
        item['name'] = row['Algorithm Name']

        # See if this alg improves the space bound
        if not best_space:
            best_space = item['space']
        elif item['space'] < best_space:
            # improvements.append(item)
            best_space = item['space']
            improvements.append(item['year'])

    return improvements

def time_improvements_by_year(family_name, variation):
    dataframe = pd.read_csv('data.csv')
    dataframe = dataframe.replace(np.nan, '', regex=True)
    algorithms = dataframe.loc[dataframe['Family Name'] == family_name]
    if variation != "by family":
        algorithms = algorithms.loc[algorithms['Variation'] == variation]
    
    if algorithms.empty:
        print('No data found for family name: ' +
              family_name + ' and variation: ' + variation)
        return False

    algorithms = algorithms[algorithms['Time Complexity Class']!=''].sort_values('Year')

    # Get the algorithms that improve the time bounds
    improvements = []
    best_time = None
    algorithms = algorithms.sort_values('Year')
    for index, row in algorithms.iterrows():
        if not best_time:
            best_time = 8 - int(row['Starting Complexity'])

        item = {}
        item['year'] = int(row['Year'])
        item['time'] = float(row['Time Complexity Class']) - 1 # math.ceil()
        item['name'] = row['Algorithm Name']

        # See if this alg improves the time bound
        if item['time'] < best_time:
            best_time = item['time']
            improvements.append(item['year'])

    return improvements

def time_improvements_by_type(family_name, variation):
    dataframe = pd.read_csv('data.csv')
    dataframe = dataframe.replace(np.nan, '', regex=True)
    algorithms = dataframe.loc[dataframe['Family Name'] == family_name]
    if variation != "by family":
        algorithms = algorithms.loc[algorithms['Variation'] == variation]
    
    if algorithms.empty:
        print('No data found for family name: ' +
              family_name + ' and variation: ' + variation)
        return False

    algorithms = algorithms[algorithms['Time Complexity Class']!=''].sort_values('Year')

    # Get the algorithms that improve the time bounds
    improvements = [] # elements of the type [a,b] where the improvements improve from runtime a to runtime b
    best_time = None
    algorithms = algorithms.sort_values('Year')
    for index, row in algorithms.iterrows():
        if not best_time:
            best_time = 8 - int(row['Starting Complexity'])

        item = {}
        item['year'] = int(row['Year'])
        item['time'] = math.ceil(float(row['Time Complexity Class']) - 1) # math.ceil()
        item['name'] = row['Algorithm Name']

        # See if this alg improves the time bound
        if item['time'] < best_time:
            improvements.append([best_time, item['time']])
            best_time = item['time']

    return improvements

def space_improvements_by_type(family_name, variation):
    dataframe = pd.read_csv('data.csv')
    dataframe = dataframe.replace(np.nan, '', regex=True)
    algorithms = dataframe.loc[dataframe['Family Name'] == family_name]
    if variation != "by family":
        algorithms = algorithms.loc[algorithms['Variation'] == variation]
    
    if algorithms.empty:
        print('No data found for family name: ' +
              family_name + ' and variation: ' + variation)
        return False

    algorithms = algorithms[algorithms['Space Complexity Class'].str.contains(':')].sort_values('Year')

    # Get the algorithms that improve the space bounds
    improvements = [] # elements of the type [a,b] where the improvements improve from runtime a to runtime b
    best_space = None
    algorithms = algorithms.sort_values('Year')
    for index, row in algorithms.iterrows():
        item = {}
        item['year'] = int(row['Year'])
        spaces = row['Space Complexity Class'].split(',\n')

        # Get the space complexity class that is relavent
        for dependency in spaces:
            dependency_list = dependency.split(': ')
            if dependency_list[0] == 'n' or dependency_list[0] == 'V':
                item['space'] = math.ceil(float(dependency_list[1]) - 1) # math.ceil(float(dependency_list[1]) - 1)
        item['space'] = item.get('space', 0)
        item['name'] = row['Algorithm Name']

        # See if this alg improves the space bound
        if not best_space:
            best_space = item['space']
        elif item['space'] < best_space:
            improvements.append([best_space, item['space']])
            best_space = item['space']

    return improvements

def plot_improvement_histograms(number_or_year, time_or_space, by_family_or_variation):
    """
    number_or_year: "Number" or "Year"
    time_or_space: "Time" or "Space"
    by_family_or_variation: "Family" or "Variation"
    """
    by_family_or_variation = by_family_or_variation.lower()

    fams = helpers.get_families()
    improvements = []
    save_dest = 'Plots/Histograms/'
    plot_title = ' Improvements '
    for fam in fams:
        if by_family_or_variation == "variation":
            vars = helpers.get_variations(fam)
        elif by_family_or_variation == "family":
            vars = ["by family"]
        for var in vars:
            if number_or_year == "Number":
                if time_or_space == "Time":
                    improvements.append(len(time_improvements_by_year(fam, var)))
                elif time_or_space == "Space":
                    improvements.append(len(space_improvements_by_year(fam, var)))
            elif number_or_year == "Year":
                if time_or_space == "Time":
                    improvements.extend(time_improvements_by_year(fam, var))
                elif time_or_space == "Space":
                    improvements.extend(space_improvements_by_year(fam, var))
    
    print(improvements)
    print(len(improvements))
    if number_or_year == "Year":
        plot_title = 'Decades ' + time_or_space + plot_title + '(by ' + by_family_or_variation + ').png'
        sns.histplot(improvements, bins=range(1940, 2030, 10))
    elif number_or_year == "Number":
        sns.histplot(improvements, bins=range(0,max(improvements) + 1,1))
        plot_title = 'Number of ' + time_or_space + plot_title + '(by ' + by_family_or_variation + ').png'
    plt.savefig(save_dest + plot_title, dpi=300, bbox_inches='tight')
    plt.clf()
    return


def plot_improvements_by_type(time_or_space, by_family_or_variation):
    save_dest = 'Plots/Heatmaps/'

    fams = helpers.get_families()
    df = pd.DataFrame(columns=['Pre-Improvement', 'Post-Improvement'])

    for fam in fams:
        if by_family_or_variation == "Variation":
            vars = helpers.get_variations(fam)
        elif by_family_or_variation == "Family":
            vars = ["by family"]

        for var in vars:
            if time_or_space == "Space":
                improvements = space_improvements_by_type(fam, var)
            elif time_or_space == "Time":
                improvements = time_improvements_by_type(fam, var)
            var_df = pd.DataFrame(improvements, columns=['Pre-Improvement', 'Post-Improvement'])
            df = pd.concat([df, var_df], ignore_index=True)
    
    # Create dataframe cross tabulation for the heatmap
    df2 = pd.crosstab(df['Pre-Improvement'], df['Post-Improvement'])
    for i in range(8):
        if i not in df2.index:
            df2.loc[i] = pd.Series(0, index=df2.columns)
    for i in range(8):
        if i not in df2.columns:
            df2[i] = pd.Series(0, index=df2.index)
    df2 = df2.sort_index(axis=0, ascending=False)
    df2 = df2.sort_index(axis=1)
    my_labels = ['constant', 'logn', 'linear', 'nlogn', 'quadratic', 'cubic', 'poly > cubic', 'exponential']
    my_labels_dict = {i: my_labels[i] for i in range(len(my_labels))}
    df2 = df2.rename(index=my_labels_dict, columns=my_labels_dict)
    df2 = df2.replace(0, np.nan)
    # print(tabulate(df2))

    # low_triang =np.rot90(np.tril(np.ones_like(df2)).astype(bool))

    ax = sns.heatmap(df2, annot=True, cmap='Greens', linewidth=0.3, vmin=0, linecolor='gray')
    plt.box(on=None)
    plt.tick_params(axis='x', colors='#A6A6A6')
    plt.tick_params(axis='y', colors='#A6A6A6')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor") # if you want x-axis label on top, change to ha="left"
    # ax.xaxis.tick_top()
    # ax.xaxis.set_label_position('top') 

    title = time_or_space + " Improvements Heatmap (by " + by_family_or_variation +")"
    plt.title(title)

    plt.tight_layout()
    

    try:
        plt.savefig(save_dest + title + ".png", dpi=300, bbox_inches='tight')
    except Exception as e:
        print(e)
        print("Failed")

    plt.clf()
    return


def plot_size_of_improvements(by_family_or_variation):
    save_dest = 'Plots/Heatmaps/'

    fams = helpers.get_families()
    space_df = pd.DataFrame(columns=['Pre-Improvement', 'Post-Improvement'])
    time_df = pd.DataFrame(columns=['Pre-Improvement', 'Post-Improvement'])

    for fam in fams:
        if by_family_or_variation == "Variation":
            vars = helpers.get_variations(fam)
        elif by_family_or_variation == "Family":
            vars = ["by family"]

        for var in vars:
            space_improvements = space_improvements_by_type(fam, var)
            space_var_df = pd.DataFrame(space_improvements, columns=['Pre-Improvement', 'Post-Improvement'])
            space_df = pd.concat([space_df, space_var_df], ignore_index=True)
            time_improvements = time_improvements_by_type(fam, var)
            time_var_df = pd.DataFrame(time_improvements, columns=['Pre-Improvement', 'Post-Improvement'])
            time_df = pd.concat([time_df, time_var_df], ignore_index=True)
    
    # Create dataframe cross tabulation for the heatmap
    space_df['Space Improvements'] = space_df['Pre-Improvement'] - space_df['Post-Improvement']
    time_df['Time Improvements'] = time_df['Pre-Improvement'] - time_df['Post-Improvement']

    df2 = pd.concat([time_df['Time Improvements'].value_counts(), space_df['Space Improvements'].value_counts()], axis=1)
    for i in range(1,8):
        if i not in df2.index:
            df2.loc[i] = pd.Series(0, index=df2.columns)
    df2 = df2.sort_index(axis=0, ascending=False)
    # print(tabulate(df2))


    ax = sns.heatmap(df2, annot=True, cmap='Greens', linewidth=0.3, vmin=0, linecolor='gray')
    plt.ylabel("Improvement Size (number of classes)")
    plt.box(on=None)
    plt.tick_params(axis='x', colors='#A6A6A6')
    plt.tick_params(axis='y', colors='#A6A6A6', rotation=0)

    title = "Size of Improvements Heatmap (by " + by_family_or_variation +")"
    plt.title(title)

    plt.tight_layout()
    # plt.show()

    try:
        plt.savefig(save_dest + title + ".png", dpi=300, bbox_inches='tight')
    except Exception as e:
        print(e)
        print("Failed")

    plt.clf()
    return


def plot_2x2_time_space_improvements(by_family_or_variation):
    save_dest = "Plots/Heatmaps/"

    dataframe = pd.read_csv('data.csv')
    dataframe = dataframe.replace(np.nan, '', regex=True)
    families = helpers.get_families()
    improves_space = []
    improves_time = []
    improves_both = []
    no_improve = []
    count = 0


    def get_array_of_improvements(algorithms, improves_space, improves_time, improves_both, no_improve, count):
        algorithms = algorithms[algorithms['Space Complexity Class'].str.contains(':')].sort_values('Year')

        best_time = 8
        best_space = 8

        for index, row in algorithms.iterrows():
            item = {}
            item['year'] = int(row['Year'])
            spaces = row['Space Complexity Class'].split(',\n')
            for dependency in spaces:
                dependency_list = dependency.split(': ')
                if dependency_list[0] == 'n' or dependency_list[0] == 'V': # TODO: maybe add a column of "preferred parameter" in the data
                    item['space'] = math.ceil(float(dependency_list[1]) - 1) # TODO: CEIL THIS (ROUND UP)
            item['space'] = item.get('space', 0)
            try:
                item['time'] = float(row['Time Complexity Class']) - 1 #math.ceil(float(row['Time Complexity Class']) - 1)
            except:
                print("Issue getting time complexity for:", row['Family Name'], row['Algorithm Name'], "\n     Time complexity class:", row['Time Complexity Class'])
                return count
            item['name'] = row['Algorithm Name']

            count += 1

            if item['space'] < best_space and item['time'] < best_time:
                best_space = item['space']
                best_time = item['time']
                improves_both.append(item['name'])
            elif item['space'] < best_space:
                best_space = item['space']
                improves_space.append(item['name'])
            elif item['time'] < best_time:
                best_time = item['time']
                improves_time.append(item['name'])
            else:
                no_improve.append(item['name'])

        return count



    for family_name in families:
        vars = helpers.get_variations(family_name)
        algorithms = dataframe.loc[dataframe['Family Name'] == family_name]
        if by_family_or_variation == "Variation":   
            for variation in vars:
                var_algorithms = algorithms.loc[algorithms['Variation'] == variation]
                if var_algorithms.empty:
                    print('No data found for family name: ' +
                        family_name + ' and variation: ' + variation)
                    continue
                count = get_array_of_improvements(var_algorithms, improves_space, improves_time, improves_both, no_improve, count)
        else:
            if algorithms.empty:
                print('No data found for family name: ' +
                    family_name)
                continue
            count = get_array_of_improvements(algorithms, improves_space, improves_time, improves_both, no_improve, count)

    print(f"Number of algorithms used: {count}\n")
    graph_df = pd.DataFrame({"Doesn't Improve": [len(no_improve)/count, len(improves_time)/count], "Improves": [len(improves_space)/count, len(improves_both)/count]},
    index=["Doesn't Improve", "Improves"])


    ax = sns.heatmap(graph_df, annot=True, cmap='Greens', linewidth=0.1, vmin=0, linecolor='gray')
    plt.ylabel("Time Improvement?")
    plt.xlabel("Space Improvement?")
    plt.box(on=None)
    plt.tick_params(axis='x', colors='#A6A6A6')
    plt.tick_params(axis='y', colors='#A6A6A6', rotation=0)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 

    title = "Proportion of Algorithms that Improve (by " + by_family_or_variation +")"
    plt.title(title)

    plt.tight_layout()
    # plt.show()

    try:
        plt.savefig(save_dest + title + ".png", dpi=300, bbox_inches='tight')
    except Exception as e:
        print(e)
        print("Failed")

    plt.clf()
    return


helpers.clean_data()
# family = 'Maximum Subarray Problem'
# variation = '1D Maximum Subarray'
# family = 'Sorting'
# variation = 'Non-Comparison Sorting'
# no_tradeoff_necessary(family, variation)

print(fraction_of_optimality())

# plot_improvement_histograms("Number", "Time", "Family")
# plot_improvement_histograms("Number", "Time", "Variation")
# plot_improvement_histograms("Number", "Space", "Family")
# plot_improvement_histograms("Number", "Space", "Variation")
# plot_improvement_histograms("Year", "Time", "Family")
# plot_improvement_histograms("Year", "Time", "Variation")
# plot_improvement_histograms("Year", "Space", "Family")
# plot_improvement_histograms("Year", "Space", "Variation")

# plot_improvements_by_type("Time", "Family")
# plot_improvements_by_type("Time", "Variation")
# plot_improvements_by_type("Space", "Family")
# plot_improvements_by_type("Space", "Variation")

# plot_size_of_improvements("Family")
# plot_size_of_improvements("Variation")

# plot_2x2_time_space_improvements("Family")
# plot_2x2_time_space_improvements("Variation")