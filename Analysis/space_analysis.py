import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import helpers
import seaborn as sns
from tabulate import tabulate
from scipy import stats

def no_tradeoff_necessary(family_name, variation):
    """
    Returns list of algorithm names if there are algorithms (among the algorithms that we have analyzed the space)
        for a problem variation that are both the fastest and most space efficient.
    Otherwise, returns []

    Prints this list as well.
    """
    df = pd.read_csv('Analysis/data.csv')
    df = df.replace(np.nan, '', regex=True)
    algorithms = df.loc[df['Family Name'] == family_name]
    algorithms = algorithms.loc[algorithms['Variation'] == variation]
    algorithms = algorithms[algorithms['Space Complexity Class'].str.contains(':')].sort_values('Year')
    
    if algorithms.empty:
        # print('No data found for family name: ' +
        #       family_name + ' and variation: ' + variation)
        return "No data"


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
            print(family_name, variation, row['Algorithm Name'], row['Time Complexity Class'])
            return "Error in Time Complexity"
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
            # total += 1
            optimal_algs = no_tradeoff_necessary(family, variation)
            if type(optimal_algs) != str:
                total += 1
                if len(optimal_algs) > 0:
                    has_optimal += 1
                    # print(family, variation, optimal_algs)
                else:
                    print(family, variation)
    return (total, has_optimal, has_optimal/total)

def space_improvements_by_year(family_name, variation):
    df = pd.read_csv('Analysis/data.csv')
    df = df.replace(np.nan, '', regex=True)
    algorithms = df.loc[df['Family Name'] == family_name]
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
    df = pd.read_csv('Analysis/data.csv')
    df = df.replace(np.nan, '', regex=True)
    algorithms = df.loc[df['Family Name'] == family_name]
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
    df = pd.read_csv('Analysis/data.csv')
    df = df.replace(np.nan, '', regex=True)
    algorithms = df.loc[df['Family Name'] == family_name]
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
    df = pd.read_csv('Analysis/data.csv')
    df = df.replace(np.nan, '', regex=True)
    algorithms = df.loc[df['Family Name'] == family_name]
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

def hist_improvements(number_or_year, time_or_space, by_family_or_variation):
    """
    number_or_year: "Number" or "Year"
    time_or_space: "Time" or "Space"
    by_family_or_variation: "Family" or "Variation"
    """
    by_family_or_variation = by_family_or_variation.lower()

    fams = helpers.get_families()
    improvements = []
    save_dest = "Analysis/Plots/Histograms/"
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
    
    if number_or_year == "Year":
        plot_title = f"Decades {time_or_space} Improvements (by {by_family_or_variation})"
        bins = range(1940, 2030, 10)
        sns.set_theme()
        counts, bins, patches = plt.hist(improvements, bins=bins)

        print(f"\n{plot_title}")
        for bar, b0, b1 in zip(counts, bins[:-1], bins[1:]):
            print(f'{b0:3d} - {b1:3d}: {bar:4.0f}')

        plt.ylabel(f"Number of {time_or_space} Improvements")
        plt.xlabel("Year")
        plt.title(f"Number of {time_or_space} Improvements per Decade (by {by_family_or_variation})")

    elif number_or_year == "Number":
        plot_title = f"Number of {time_or_space} Improvements (by {by_family_or_variation})"
        bins = range(0,max(improvements) + 1,1)
        labels, counts = np.unique(improvements, return_counts=True)
        sns.set_theme()
        ax = plt.bar(labels, counts, align='center')
        plt.gca().set_xticks(labels)

        print(f"\n{plot_title}")
        for container in ax:
            bar_x = int(container.get_x() + container.get_width() / 2)
            bar_y = container.get_height()
            print(f'{bar_x:3d}: {bar_y:4.0f}')
            
        plt.ylabel(f"Number of Problems")
        plt.xlabel(f"Number of {time_or_space} Improvements")
        plt.title(f"Number of Problems With a Given Number\nof {time_or_space} Improvements (by {by_family_or_variation})")
        xint = range(0, math.ceil(max(improvements))+1)
        plt.xticks(xint)
    plt.savefig(save_dest + plot_title + ".png", dpi=300, bbox_inches="tight")
    plt.clf()
    plt.close("all")
    return

def heat_improvements_by_type(time_or_space, by_family_or_variation):
    save_dest = 'Analysis/Plots/Heatmaps/'

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
    
    # Create df cross tabulation for the heatmap
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
    plt.close("all")
    return

def heat_size_of_improvements(by_family_or_variation):
    save_dest = 'Analysis/Plots/Heatmaps/'

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
    
    # Create df cross tabulation for the heatmap
    space_df['Space Improvements'] = space_df['Pre-Improvement'] - space_df['Post-Improvement']
    time_df['Time Improvements'] = time_df['Pre-Improvement'] - time_df['Post-Improvement']

    df2 = pd.concat([time_df['Time Improvements'].value_counts(), space_df['Space Improvements'].value_counts()], axis=1)
    for i in range(1,8):
        if i not in df2.index:
            df2.loc[i] = pd.Series(0, index=df2.columns)
    df2 = df2.sort_index(axis=0, ascending=False)
    # print(tabulate(df2))


    ax = sns.heatmap(df2, annot=True, cmap='Greens', linewidth=0.3, vmin=0, linecolor='gray', fmt='.3g')
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
    plt.close("all")
    return

def heat_2x2_time_space_improvements(by_problem_or_algorithm, by_family_or_variation, include_first_algo):
    _not = "Not " if not include_first_algo else ""
    save_dest = f"Analysis/Plots/Heatmaps/Proportions {_not}Including Firsts/"

    df = pd.read_csv('Analysis/data.csv')
    df = df.replace(np.nan, '', regex=True)
    families = helpers.get_families()
    improves_space = []
    improves_time = []
    improves_both = []
    no_improve = []
    count = 0


    def get_array_of_improvements(algorithms, improves_space, improves_time, improves_both, no_improve, count, include_first_algo):
        algorithms = algorithms[algorithms['Space Complexity Class'].str.contains(':')].sort_values('Year')

        best_time = 8
        best_space = 8
        first_algo = not include_first_algo
        improved_space = False
        improved_time = False

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

            if by_problem_or_algorithm == "Algorithm":
                if first_algo:
                    first_algo = False
                    if item['space'] < best_space and item['time'] < best_time:
                        best_space = item['space']
                        best_time = item['time']
                    elif item['space'] < best_space:
                        best_space = item['space']
                    elif item['time'] < best_time:
                        best_time = item['time']
                else:
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
            elif by_problem_or_algorithm == "Problem":
                if first_algo:
                    first_algo = False
                    if item['space'] < best_space and item['time'] < best_time:
                        best_space = item['space']
                        best_time = item['time']
                    elif item['space'] < best_space:
                        best_space = item['space']
                    elif item['time'] < best_time:
                        best_time = item['time']
                else:
                    if item['space'] < best_space and item['time'] < best_time:
                        improved_space = True
                        improved_time = True
                    elif item['space'] < best_space:
                        improved_space = True
                    elif item['time'] < best_time:
                        improved_time = True

                    if improved_time and improved_space:
                        break

        if by_problem_or_algorithm == "Problem":
            count += 1
            if improved_time and improved_space:
                improves_both.append(1)
            elif improved_time:
                improves_time.append(1)
            elif improved_space:
                improves_space.append(1)
            else:
                no_improve.append(1)

        return count



    for family_name in families:
        vars = helpers.get_variations(family_name)
        algorithms = df.loc[df['Family Name'] == family_name]
        if by_family_or_variation == "Variation":   
            for variation in vars:
                var_algorithms = algorithms.loc[algorithms['Variation'] == variation]
                if var_algorithms.empty:
                    print('No data found for family name: ' +
                        family_name + ' and variation: ' + variation)
                    continue
                count = get_array_of_improvements(var_algorithms, improves_space, improves_time, improves_both, no_improve, count, include_first_algo)
        else:
            if algorithms.empty:
                print('No data found for family name: ' +
                    family_name)
                continue
            count = get_array_of_improvements(algorithms, improves_space, improves_time, improves_both, no_improve, count, include_first_algo)

    # print(f"Number of {by_problem_or_algorithm}s used: {count}\n")
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

    title = f"Proportion of {by_problem_or_algorithm}s that Improve (by {by_family_or_variation})"
    plt.title(title)

    plt.tight_layout()

    try:
        plt.savefig(save_dest + title + ".png", dpi=300, bbox_inches='tight')
    except Exception as e:
        print(e)
        print("Failed")

    plt.clf()
    plt.close("all")
    return

def pie_best_space(num_for_split, by_family_or_variation):
    """
    0: Constant
    1: Log n
    2: Linear
    ...

    Args:
        num_for_split: The number of complexity class that you want split at (e.g. sub___, ___, super___)

    Returns:
        List of two tuples:
            Tuple of three ints (# problems sub___, # problems ___, # problems super___)
            Tuple of the proportions of the above
    """

    df = pd.read_csv('Analysis/data.csv')
    df = df.replace(np.nan, '', regex=True)
    families = helpers.get_families()
    sub_space = 0
    at_space = 0
    super_space = 0
    count = 0

    class_string = {0: "Constant", 1: "Logarithmic", 2: "Linear", 3: "n log n", 4: "Quadratic", 5: "Cubic", 6: "Polynomial (> 3)", 7: "Exponential"}

    def get_best_space(algorithms):
        """Get the best space complexity from the input algorithms (e.g. from a family or variation)"""
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
            item['name'] = row['Algorithm Name']

            if item['space'] < best_space:
                best_space = item['space']

        return best_space

    if by_family_or_variation == "Variation":
        print(f"\nProblem variations with best-space that is super-{class_string[num_for_split]}")
    else:
        print(f"\nProblem families with best-space that is super-{class_string[num_for_split]}")

    for family_name in families:
        vars = helpers.get_variations(family_name)
        algorithms = df.loc[df['Family Name'] == family_name]
        if by_family_or_variation == "Variation":   
            for variation in vars:
                var_algorithms = algorithms.loc[algorithms['Variation'] == variation]
                if var_algorithms.empty:
                    print('No data found for family name: ' +
                        family_name + ' and variation: ' + variation)
                    continue
                best_space = get_best_space(var_algorithms)
                count += 1
                if best_space < num_for_split:
                    sub_space += 1
                elif best_space == num_for_split:
                    at_space += 1
                elif best_space > num_for_split:
                    print(f"{family_name}: {variation} -- {class_string[best_space]}")
                    super_space += 1
        else:
            if algorithms.empty:
                print('No data found for family name: ' +
                    family_name)
                continue
            best_space = get_best_space(algorithms)
            count += 1
            if best_space < num_for_split:
                sub_space += 1
            elif best_space == num_for_split:
                at_space += 1
            elif best_space > num_for_split:
                print(f"{family_name} -- {class_string[best_space]}")
                super_space += 1
                
    print(f"\nNumbers of {by_family_or_variation}:")
    print(f"Sub-{class_string[num_for_split]}: {sub_space}, ({sub_space / count})")
    print(f"At-{class_string[num_for_split]}: {at_space}, ({at_space / count})")
    print(f"Super-{class_string[num_for_split]}: {super_space}, ({super_space / count})\n")
    if num_for_split == 0:
        data = [at_space, super_space]
        labels = [f"At-{class_string[num_for_split]}", f"Super-{class_string[num_for_split]}"]
        explode = [0, 0]
    else:
        data = [sub_space, at_space, super_space]
        labels = [f"Sub-{class_string[num_for_split]}", f"At-{class_string[num_for_split]}", f"Super-{class_string[num_for_split]}"]
        explode = [0,0,0.1]
    sns.set_palette("Reds", 3)
    plt.figure(figsize=(18, 9))
    plt.pie(data, labels=labels, autopct='%1.1f%%', explode=explode)
    plt.title(f"Problems' Best Space Complexity Compared to {class_string[num_for_split]} Space")
    save_dest = "Analysis/Plots/Best Space Algos/"
    plt.savefig(f"{save_dest}{class_string[num_for_split]} Pie (by {by_family_or_variation}).png", dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close("all")
    return [(sub_space, at_space, super_space), (sub_space / count, at_space / count, super_space / count)]

def hist_papers_per_decade():
    """
    Create a histogram showing the number of papers in our database per decade.

    Note: "1940s" is actually "1940s and before"
    """

    df = pd.read_csv('Analysis/data_dirty.csv')
    df = df.replace(np.nan, '', regex=True)

    save_dest = "Analysis/Plots/Histograms/"
    plot_title = "Number of Papers Per Decade"

    df.loc[df["Year"] < 1940] = 1940
    years = df[df["Derived Space Complexity?"].isin(["0", "1"])]["Year"]
    bins = range(1940, 2030, 10)
    sns.set_theme()
    counts, bins, patches = plt.hist(years, bins=bins)

    print(f"{plot_title}")
    for bar, b0, b1 in zip(counts, bins[:-1], bins[1:]):
        print(f'{b0:3d} - {b1:3d}: {bar:4.0f}')

    plt.ylabel("Number of Papers")
    plt.xlabel("Decade Published")
    plt.title(plot_title)
    plt.tight_layout()

    plt.savefig(save_dest + plot_title + ".png", dpi=300, bbox_inches="tight")
    plt.clf()
    plt.close("all")
    return

def hist_space_analysis_per_decade():
    """
    Creates two histograms--one showing the number of papers in our database that actually analyzed space complexity per decade,
    and one showing the number of papers that didn't analyze the space complexity (and we had to derive it ourselves) per decade.

    Note: "1940s" is actually "1940s and before"
    """

    df = pd.read_csv('Analysis/data_dirty.csv')
    df = df.replace(np.nan, '', regex=True)

    save_dest = "Analysis/Plots/Histograms/"

    # Papers With Space Complexity Analysis
    plot_title = "Number of Papers Per Decade With Space Analysis"

    df.loc[df["Year"] < 1940] = 1940
    analyzed_years = df[df["Derived Space Complexity?"] == "0"]["Year"]
    analyzed_years = analyzed_years.rename("With Space Analysis")

    bins = range(1940, 2030, 10)
    sns.set_theme()
    counts, bins, patches = plt.hist(analyzed_years, bins=bins)

    print(f"\n{plot_title}")
    for bar, b0, b1 in zip(counts, bins[:-1], bins[1:]):
        print(f'{b0:3d} - {b1:3d}: {bar:4.0f}')

    plt.ylabel("Number of Papers")
    plt.xlabel("Decade Published")
    plt.title(plot_title)
    plt.tight_layout()

    plt.savefig(save_dest + plot_title + ".png", dpi=300, bbox_inches="tight")
    plt.clf()
    plt.close("all")

    # Papers Without Space Complexity Analysis
    plot_title = "Number of Papers Per Decade Without Space Analysis"

    df.loc[df["Year"] < 1940] = 1940
    not_analyzed_years = df[df["Derived Space Complexity?"] == "1"]["Year"]
    not_analyzed_years = not_analyzed_years.rename("Without Space Analysis")
    
    bins = range(1940, 2030, 10)
    sns.set_theme()
    counts, bins, patches = plt.hist(not_analyzed_years, bins=bins)

    print(f"\n{plot_title}")
    for bar, b0, b1 in zip(counts, bins[:-1], bins[1:]):
        print(f'{b0:3d} - {b1:3d}: {bar:4.0f}')

    plt.ylabel("Number of Papers")
    plt.xlabel("Decade Published")
    plt.title(plot_title)
    plt.tight_layout()

    plt.savefig(save_dest + plot_title + ".png", dpi=300, bbox_inches="tight")
    plt.clf()
    plt.close("all")

    # Putting the two side-by-side
    plot_title = "Space Analysis Per Decade (Bars)"

    plt.ylabel("Number of Papers")
    plt.xlabel("Decade Published")
    
    bins = range(1940, 2030, 10)
    sns.set_theme()
    counts, bins, patches = plt.hist([analyzed_years, not_analyzed_years], bins=bins,
                                    label=["With Space Analysis", "Without Space Analysis"],
                                    histtype="bar")

    with_space, without_space = counts
    print(f"\n{plot_title} -- Papers With vs Without Space Analysis")
    for with_bar, without_bar, b0, b1 in zip(with_space, without_space, bins[:-1], bins[1:]):
        print(f'{b0:3d} - {b1:3d}: {with_bar:4.0f} {without_bar:4.0f}')

    plt.title(plot_title)
    plt.legend(loc="upper left")
    plt.tight_layout()

    plt.savefig(save_dest + plot_title + ".png", dpi=300, bbox_inches="tight")
    plt.clf()
    plt.close("all")
    return

def hist_space_and_time_improvements_per_decade(by_family_or_variation):
    by_family_or_variation = by_family_or_variation.lower()

    fams = helpers.get_families()
    space_improvements = []
    time_improvements = []

    for fam in fams:
        if by_family_or_variation == "variation":
            vars = helpers.get_variations(fam)
        elif by_family_or_variation == "family":
            vars = ["by family"]
        for var in vars:
            space_improvements.extend(space_improvements_by_year(fam, var))
            time_improvements.extend(time_improvements_by_year(fam, var))

    # Putting the two side-by-side
    save_dest = "Analysis/Plots/Histograms/"
    plot_title = "Improvements Per Decade (Bars)"
    
    bins = range(1940, 2030, 10)
    sns.set_theme()
    counts, bins, patches = plt.hist([time_improvements, space_improvements], bins=bins,
                                    label=["Time Improvements", "Space Improvements"],
                                    histtype="bar")

    time, space = counts
    print(f"\n{plot_title} -- Time improvements vs space improvements")
    for time_bar, space_bar, b0, b1 in zip(time, space, bins[:-1], bins[1:]):
        print(f'{b0:3d} - {b1:3d}: {time_bar:4.0f} {space_bar:4.0f}')

    plt.ylabel("Number of Improvements")
    plt.xlabel("Decade Published")
    plt.title(plot_title)
    plt.legend(loc="upper left")
    plt.tight_layout()

    plt.savefig(save_dest + plot_title + ".png", dpi=300, bbox_inches="tight")
    plt.clf()
    plt.close("all")
    return


helpers.clean_data()
# family = 'Maximum Subarray Problem'
# variation = '1D Maximum Subarray'
# family = 'Sorting'
# variation = 'Non-Comparison Sorting'
# no_tradeoff_necessary(family, variation)

print(fraction_of_optimality())

print("\n-----------------------------\n")

hist_improvements("Number", "Time", "Family")
hist_improvements("Number", "Time", "Variation")
hist_improvements("Number", "Space", "Family")
hist_improvements("Number", "Space", "Variation")
hist_improvements("Year", "Time", "Family")
hist_improvements("Year", "Time", "Variation")
hist_improvements("Year", "Space", "Family")
hist_improvements("Year", "Space", "Variation")

print("\n-----------------------------\n")

heat_improvements_by_type("Time", "Family")
heat_improvements_by_type("Time", "Variation")
heat_improvements_by_type("Space", "Family")
heat_improvements_by_type("Space", "Variation")

print("\n-----------------------------\n")

heat_size_of_improvements("Family")
heat_size_of_improvements("Variation")

print("\n-----------------------------\n")

heat_2x2_time_space_improvements("Algorithm", "Family", include_first_algo=True)
heat_2x2_time_space_improvements("Algorithm", "Variation", include_first_algo=True)
heat_2x2_time_space_improvements("Problem", "Family", include_first_algo=True)
heat_2x2_time_space_improvements("Problem", "Variation", include_first_algo=True)

print("\n-----------------------------\n")

pie_best_space(0, "Family")
pie_best_space(0, "Variation")
pie_best_space(2, "Family")
pie_best_space(2, "Variation")

print("\n-----------------------------\n")

hist_papers_per_decade()
hist_space_analysis_per_decade()
hist_space_and_time_improvements_per_decade("Variation")