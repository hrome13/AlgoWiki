import math
from os import truncate
from tkinter import E
from turtle import title
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import helpers
import seaborn as sns


def generate_graph(family_name, variation, type):
    """
    type: one of ['time', 'space', 'both_time_improvements', 'both_tradeoffs', 'pareto_decades']
    """
    dataframe = pd.read_csv('Analysis/data.csv')
    dataframe = dataframe.replace(np.nan, '', regex=True)

    # Get the algorithms for the specific family/variation
    algorithms = dataframe.loc[dataframe['Family Name'] == family_name].sort_values("Year")


    if variation != "By Family":
        # Set the starting algo for the variation as the first algo for the family (typically the naive algo)
        algorithms.loc[:, "Starting Time"] = algorithms["Time Complexity Class"].iloc[0]
        algorithms.loc[:, "Starting Year"] = algorithms["Year"].iloc[0]
        algorithms.loc[:, "Starting Name"] = algorithms["Algorithm Name"].iloc[0]
        row = algorithms.iloc[0]
        spaces = row['Space Complexity Class'].split(',\n')
        starting_space = 0
        for dependency in spaces:
            dependency_list = dependency.split(': ')
            if dependency_list[0] == 'n' or dependency_list[0] == 'V':
                starting_space = float(dependency_list[1]) - 1
        algorithms.loc[:, "Starting Space"] = starting_space

        # Filter by variation name
        algorithms = algorithms.loc[algorithms['Variation'] == variation]

    sns.set_theme()
    
    # Check that the problem (+ variation) has data
    if algorithms.empty:
        print('No data found for family name: ' +
              family_name + ' and variation: ' + variation)
        return False
    if type == 'time':
        algorithms = algorithms[algorithms['Time Complexity Class']!=''].sort_values('Year')
        if algorithms.empty:
            print('No time complexity data found for family name: ' +
              family_name + ' and variation: ' + variation)
            return False
    elif type == 'both_time_improvements':
        algorithms = algorithms[algorithms['Transition Class'].str.contains('->')].sort_values('Transition Class')
    elif type == 'space' or type == 'both_tradeoffs' or type == 'pareto_decades':
        algorithms = algorithms[algorithms['Space Complexity Class'].str.contains(':')].sort_values('Year')
        if algorithms.empty:
            print('No space complexity data found for family name: ' +
              family_name + ' and variation: ' + variation)
            return False

    if type in {'both_time_improvements', 'both_tradeoffs'}: #time
        upper_start_complexity = algorithms['Starting Complexity'].iloc[0]

        upper_bound = [
            {
                'year': 1940,
                'time': 8 - int(upper_start_complexity),
                'space': 8 - int(upper_start_complexity)
            },
        ]

        for index, row in algorithms.iterrows():
            item = {}
            item['year'] = int(row['Year'])

            spaces = row['Space Complexity Class'].split(',\n')
            for dependency in spaces:
                dependency_list = dependency.split(': ')
                if dependency_list[0] == 'n':
                    item['space'] = float(dependency_list[1]) - 1
            item['space'] = item.get('space', 0)

            if type == 'time' or type == 'both_time_improvements':
                item['time'] = 8 - int(row['Transition Class'][3])
            else:
                item['time'] = int(row['Time Complexity Class']) - 1
            item['name'] = row['Algorithm Name']
            upper_bound.append(item)

        upper_bound.append({
            'year': 2021,
            'time': upper_bound[-1]['time'],
            'space': upper_bound[-1]['space']
        })

        upper_x = [x['year'] for x in upper_bound]
        upper_time = [x['time'] for x in upper_bound]
        upper_space = [x['space'] for x in upper_bound]

        lower_x = [1940, 2021]
        lower_time = [2, 2]

    plt.figure(figsize=(18, 9))
    save_dest = ''

    # Create time bound improvement step plots
    if type == 'time':
        if variation == "By Family":
            save_dest = "Analysis/Plots/Time Improvements/By Family/"
        else:
            save_dest = "Analysis/Plots/Time Improvements/By Variation/"

        # Get the algorithms that improve the time bounds
        improvements = []
        best_time = None
        improves = False
        algorithms = algorithms.sort_values('Year')
        for index, row in algorithms.iterrows():

            # Add the first algo for the problem family as the starting point for the variation
            # if best_time is None and var != "By Family":
            #     if row["Starting Year"] != row["Year"] or row["Starting Name"] != row["Algorithm Name"] or row["Starting Time"] != row["Time Complexity Class"]:
            #         item = {'year': int(row["Starting Year"]), 'time': float(row["Starting Time"]) - 1, 'name': row["Starting Name"]}
            #         best_time = item["time"]
            #         improvements.append(item)
            
            item = {'year': int(row["Year"]), 'time': float(row["Time Complexity Class"]) - 1, 'name': row["Algorithm Name"]}

            # See if this algo improves the time bound
            if best_time is None:
                improvements.append(item)
                best_time = item['time']
            else:
                if item['time'] < best_time:
                    improvements.append(item)
                    best_time = item['time']
                    improves = True

        if improves:
            save_dest += "Improvements/"
        else:
            save_dest += "No Improvements/"

        # Gather the improvement algs into useful lists
        time_years = [alg['year'] for alg in improvements]
        time_class = [alg['time'] for alg in improvements]
        time_labels = [alg['name'] for alg in improvements]

        # Need this for the step plot
        time_years.append(2021)
        time_class.append(time_class[-1])
        time_labels.append('')

        # Plot and annotate the algs that improve space bounds
        plt.step(time_years, time_class, where='post', color="#822600",
                 linewidth=5, solid_capstyle='round')
        plt.plot(time_years[:-1], time_class[:-1], 'o', color='#C71F1D', markersize=14)
        # plt.annotate(time_labels[0].split(';')[-1], xy=(time_years[0], time_class[0]),
        #              xytext=(time_years[0] + 1, time_class[0] + 0.15), color="brown", fontsize=12, weight='bold')
        for i in range(0, len(time_labels) - 1):
            plt.annotate(time_labels[i].split(';')[-1] + ', ' + str(time_years[i]), xy=(time_years[i], time_class[i]),
                        xytext=(time_years[i] + 1, time_class[i] + 0.15), color='#C71F1D', fontsize=12, weight='bold')# , rotation=45)
        title_suffix = 'Time'

    # Create space bound improvement step plots
    elif type == 'space':
        if variation == "By Family":
            save_dest = "Analysis/Plots/Space Improvements/By Family/"
        else:
            save_dest = "Analysis/Plots/Space Improvements/By Variation/"

        # Get the algorithms that improve the space bounds
        improvements = []
        best_space = None
        improves = False
        algorithms = algorithms.sort_values('Year')
        for index, row in algorithms.iterrows():

            # Add the first algo for the problem family as the starting point for the variation
            # if best_space is None and var != "By Family":
            #     if row["Starting Year"] != row["Year"] or row["Starting Name"] != row["Algorithm Name"] or row["Starting Time"] != row["Time Complexity Class"]:
            #         item = {'year': int(row["Starting Year"]), 'space': float(row["Starting Space"]) - 1, 'name': row["Starting Name"]}
            #         best_space = item["space"]
            #         improvements.append(item)

            item = {}
            item['year'] = int(row['Year'])
            spaces = row['Space Complexity Class'].split(',\n')

            # Get the space complexity class that is relavent
            for dependency in spaces:
                dependency_list = dependency.split(': ')
                if dependency_list[0] == 'n' or dependency_list[0] == 'V':
                    item['space'] = float(dependency_list[1]) - 1
            item['space'] = item.get('space', 0)
            item['name'] = row['Algorithm Name']

            # See if this algo improves the space bound
            if best_space is None:
                improvements.append(item)
                best_space = item['space']
            else:
                if item['space'] < best_space:
                    improvements.append(item)
                    best_space = item['space']
                    improves = True

        if improves:
            save_dest += "Improvements/"
        else:
            save_dest += "No Improvements/"

        # Gather the improvement algs into useful lists
        space_years = [alg['year'] for alg in improvements]
        space_class = [alg['space'] for alg in improvements]
        space_labels = [alg['name'] for alg in improvements]

        # Need this for the step plot
        space_years.append(2021)
        space_class.append(space_class[-1])
        space_labels.append('')

        # Plot and annotate the algs that improve space bounds
        plt.step(space_years, space_class, where='post', color="#822600",
                 linewidth=5, solid_capstyle='round')
        plt.plot(space_years[:-1], space_class[:-1], 'o', color='#C71F1D', markersize=14)
        # plt.annotate(space_labels[0].split(';')[-1], xy=(space_years[0], space_class[0]),
        #              xytext=(space_years[0] + 1, space_class[0] + 0.15), color="brown", fontsize=12, weight='bold')
        for i in range(0, len(space_labels) - 1):
            plt.annotate(space_labels[i].split(';')[-1] + ', ' + str(space_years[i]), xy=(space_years[i], space_class[i]),
                        xytext=(space_years[i] + 1, space_class[i] + 0.15), color='#C71F1D', fontsize=12, weight='bold') #, rotation=45)
        title_suffix = 'Space'

    # Create scatterplot showing space vs time for the algos that improve the time upper bounds
    elif type == 'both_time_improvements':
        plt.step(upper_x, upper_time, where='post', color="#822600",
                linewidth=5, solid_capstyle='round', label='Time')
        plt.annotate('Brute Force', xy=(upper_x[0], upper_time[0]), xytext=(
            upper_x[0], upper_time[0] + 0.15), color='#822600', fontsize=12, weight='bold')

        plt.plot(upper_x[1:-1], upper_time[1:-1], 'o', color='#C71F1D', markersize=14)
        for i in range(1, len(upper_x) - 1):
            plt.annotate(upper_bound[i]['name'].split(';')[-1] + ', ' + str(upper_bound[i]['year']), xy=(upper_x[i], upper_time[i]),
                        xytext=(upper_x[i] + 1, upper_time[i] + 0.15), color='#C71F1D', fontsize=12, weight='bold')

        plt.step(upper_x, upper_space, where='post', color="#FA9A20",
                    linewidth=5, solid_capstyle='round', alpha=0.5, label='Space')
        plt.plot(upper_x[1:-1], upper_space[1:-1], 'o', color='#C71F1D', markersize=14)

        plt.legend(prop={'size': 24})
        title_suffix = 'Time with Space'

    # Create scatterplot showing space vs time for all of the algos
    elif type == 'both_tradeoffs':
        plt.plot(upper_space, upper_time, 'o', color='#C71F1D', markersize=14)
        plt.xlabel('Space Complexity (Auxiliary)')
        plt.ylabel('Time Complexity')
        # for i in range(1, len(upper_x) - 1):
        #     plt.annotate(upper_bound[i]['name'].split(';')[-1] + ', ' + str(upper_bound[i]['year']), xy=(upper_space[i], upper_time[i]),
        #                 xytext=(upper_x[i] + 1, upper_space[i] + 0.15), color='#C71F1D', fontsize=12, weight='bold', rotation=45)
        title_suffix = 'Tradeoffs'

    # Create plot showing the Pareto fronteir at different decades
    elif type == 'pareto_decades':
        decade_start = (algorithms['Year'].min() // 10) * 10
        decade_algs = {decade_start: []}

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
            item['quantum?'] = row['Quantum?'] == 1 or row['Quantum?'] == "1"
            item['parallel?'] = row['Parallel?'] == 1 or row['Parallel?'] == "1"
            item['space'] = item.get('space', 0)
            item['name'] = row['Algorithm Name']
            item['time'] = float(row['Time Complexity Class']) - 1 # ceil?

            # See if this alg is in the current decade, else update the decade (and plot the previous decade's algs)
            if item['year'] < decade_start + 10:
                decade_algs[decade_start].append(item)
            else:
                last_decade = decade_start
                decade_start = (item['year'] // 10) * 10
                decade_algs[decade_start] = decade_algs[last_decade].copy()
                decade_algs[decade_start].append(item)

        frontier = {} # mapping time -> best space for that time
        frontier_algs = {}
        decades = {}

        for decade in decade_algs:
            label = str(decade) + '\'s'

            for item in decade_algs[decade]:
                on_frontier = True
                # for time in range(0, item['time']):
                    # if time in frontier:
                    #     if item['space'] >= frontier[time]:
                    #         on_frontier = False
                    #         break
                for time in frontier:
                    if time < item['time']:
                        if item['space'] >= frontier[time]:
                            on_frontier = False
                            break
                if on_frontier:
                    if item['time'] in frontier:
                        if item['space'] < frontier[item['time']]:
                            frontier_algs[item['time']]= [item]
                            frontier[item['time']] = item['space']
                        elif item['space'] == frontier:
                            frontier_algs[item['time']].append(item)
                    else:
                        frontier[item['time']] = item['space']
                        frontier_algs[item['time']] = [item]

            spaces = []
            times = []
            names = []
            years = []
            parallels = []
            quantums = []
            prev_decades = decades.keys()

            for time in frontier_algs:
                alg = frontier_algs[time][0]
                skip = False # turns to True if it improves the Pareto frontier

                # Compare algo to previous Pareto frontier
                for x in prev_decades:
                    if decades[x][0]:
                        for prev_sp, prev_ti in zip(decades[x][0], decades[x][1]):
                            if prev_sp == alg['space'] and prev_ti == alg['time']:
                                skip = True
                                break
                        if skip:
                            break

                # If the algo improves the Pareto frontier, add it to the lists
                if not skip:
                    parallels.append(alg['parallel?'])
                    quantums.append(alg['quantum?'])
                    spaces.append(alg['space'])
                    times.append(alg['time'])
                    names.append(alg['name'])
                    years.append(alg['year'])

            decades[decade] = (spaces, times, names, years, parallels, quantums)

        decades_list = [x for x in decades]
        decades_list.sort()
        points_plotted = 0

        # Drop the decade colors
        colors = sns.color_palette("Reds", n_colors=4)#len(decades))
        colors_idx = 3
        # colors_idx = len(decades) - 1

        # Draw arrows going chronologically first to second, second to third, etc.
        all_spaces = []
        all_times = []
        for decade in decades_list:
            spaces, times, names, years, parallels, quantums = decades[decade]
            together = [(space_, time_, year_) for space_, time_, year_ in zip(spaces, times, years)]
            together = sorted(together, key=lambda x: x[2])
            spaces = [x[0] for x in together]
            times = [x[1] for x in together]
            years = [x[2] for x in together]
            all_spaces.extend(spaces)
            all_times.extend(times)
        marker_radius = 0.045
        for x1, x2, y1, y2 in zip(all_spaces[:-1], all_spaces[1:], all_times[:-1], all_times[1:]):
            if family_name == "Matrix Product" and x1 != 0:
                continue
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                y_off = marker_radius + 0.02
                x_off = 0
            else:
                theta = np.arctan(dy / dx)
                x_off = marker_radius * np.cos(theta)
                y_off = marker_radius * np.sin(theta)
                if dx > 0:
                    x_off *= -1
                    y_off *= -1
            plt.annotate("",
                         xy=(x1 - x_off, y1 - y_off), xycoords='data',
                         xytext=(x2 + x_off, y2 + y_off), textcoords='data',
                         arrowprops=dict(arrowstyle="<|-",
                                         color="black", lw=1.75),
                        )

        for decade in decades_list[::-1]:
            label = str(decade) + 's'
            spaces, times, names, years, parallels, quantums = decades[decade]
            if spaces and times:
                color = colors[colors_idx]
                # plt.step(spaces, times, label=label, where='post', linewidth=5, c=c)

                spaces = np.array(spaces)
                times = np.array(times)
                parallels = np.array(parallels)
                quantums = np.array(quantums)

                # Plot the quantum algos
                if sum(quantums) > 0:
                    plt.plot(spaces[quantums == True], times[quantums == True], 'X', markersize=14, label=label+" (Quantum)", color=color)

                # Plot the parallel algos
                if sum(parallels) > 0:
                    plt.plot(spaces[parallels == True], times[parallels == True], 'P', markersize=14, label=label+" (Parallel)", color=color)

                # Plot the rest
                if len(spaces[(parallels == False) & (quantums == False)]) > 0:
                    plt.plot(spaces[(parallels == False) & (quantums == False)],
                            times[(parallels == False) & (quantums == False)],
                            'o', markersize=14, label=label, color=color)
                
                points_plotted += len(names)

                # Annotate the algos
                if family_name == "Matrix Product":
                    if decade == decades_list[0]:
                        i = -1
                        plt.annotate(names[i].split(';')[-1] + ', ' + str(years[i]), xy=(spaces[i], times[i]),
                                    xytext=(spaces[i] + 0.15, times[i] + 0.15), color='#C71F1D', fontsize=12, weight='bold', rotation=0)
                    elif decade == decades_list[-1]:
                        i = -1
                        plt.annotate(names[i].split(';')[-1] + ', ' + str(years[i]), xy=(spaces[i], times[i]),
                                    xytext=(spaces[i] + 0.15, times[i]), color='#C71F1D', fontsize=12, weight='bold', rotation=0)
                else:
                    for i in range(len(names)):
                        plt.annotate(names[i].split(';')[-1] + ', ' + str(years[i]), xy=(spaces[i], times[i]),
                                    xytext=(spaces[i] + 0.15, times[i] + 0.15), color='#C71F1D', fontsize=12, weight='bold', rotation=0)#10)
                
            # Drop the decade colors
            # colors_idx -= 1
        
        # Drop the legend
        # plt.legend(prop={'size': 24})

        plt.xlabel("Space Complexity (Auxiliary)", fontsize=20)
        plt.ylabel("Time Complexity", fontsize=20)
        if variation == "By Family":
            save_dest = "Analysis/Plots/Pareto Decades/By Family/"
        else:
            save_dest = "Analysis/Plots/Pareto Decades/By Variation/"

        if points_plotted > 1:
            save_dest += "Improvements/"
        else:
            save_dest += "No Improvements/"
        title_suffix = "Pareto Frontier"

    plt.box(on=None)
    plt.tick_params(axis='x', colors='Black', which='both', bottom=False, top=False, labelbottom=True)
    plt.tick_params(axis='y', colors='Black')

    if variation != "" and variation != "By Family" and variation != family_name:
        plt.title(variation + ' ('+family_name+')',
                  fontsize=20, color='Black') # #A6A6A6
    else:
        plt.title(family_name, fontsize=20, color='Black')

    # plt.yticks([0, 1, 2, 3, 4, 5, 6, 7], ('constant', 'logn', 'linear',
    #                                       'nlogn', 'quadratic', 'cubic', 'poly > cubic', 'exponential'), weight='bold', fontsize=14)
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7], ('Constant', 'Logarithmic', 'Linear',
                                          'n log n', 'Quadratic', 'Cubic', 'Poly (> Cubic)', 'Exponential'), weight='bold', fontsize=14)
    if type == 'both_tradeoffs' or type == 'pareto_decades':
        # plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], ('constant', 'logn', 'linear',
        #                                   'nlogn', 'quadratic', 'cubic', 'poly > cubic', 'exponential'), weight='bold', fontsize=14)
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], ('Constant', 'Logarithmic', 'Linear',
                                          'n log n', 'Quadratic', 'Cubic', 'Poly (> Cubic)', 'Exponential'), weight='bold', fontsize=14)
        plt.grid(axis='both', color='#E8E8E8')
    else:
        plt.xticks(weight='bold', fontsize=14)
        plt.grid(axis='y', color='#E8E8E8')
    plt.tight_layout()

    try:
        if variation != "" and variation != "By Family":
            save_title = f"{family_name} - {variation} - {title_suffix}.png"
        else:
            save_title = f"{family_name} - {title_suffix}.png"
        plt.savefig(save_dest+save_title, dpi=300, bbox_inches='tight')
    except Exception as e:
        print(e)
        print("Failed")
    plt.clf()
    plt.close("all")


helpers.clean_data()

# family = 'Maximum Subarray Problem'
# variation = '1D Maximum Subarray'
# family = 'Optimal Binary Search Trees'
# variation = 'Approximate OBST'
# family = 'Sorting'
# variation = 'Non-Comparison Sorting'
# family = 'Sorting'
# variation = 'Comparison Sorting'
# family = "De Novo Genome Assembly"
# variation = "De Novo Genome Assembly"
# family = "Integer Factoring"
# variation = "Second Category Integer Factoring"
# family = "Matrix Product"
# variation = "Matrix Multiplication"
# family = "Motif Search"
# variation = "Motif Search"
family = "Sorting"
variation = "Comparison Sorting"

generate_graph(family, variation, 'space')
generate_graph(family, variation, 'time')
generate_graph(family, variation, 'pareto_decades')
# generate_graph(family, variation, 'both_time_improvements')
# generate_graph(family, variation, 'both_tradeoffs')

# families = helpers.get_families()
# for fam in families:
#     for var in helpers.get_variations(fam):
#         generate_graph(fam, var, 'pareto_decades')
        # generate_graph(fam, var, 'space')
        # generate_graph(fam, var, 'time')
        # try:
            # generate_graph(fam, var, 'pareto_decades')
            # generate_graph(fam, var, 'space')
            # generate_graph(fam, var, 'time')
        # except Exception as e:
        #     print(fam, var, e)


# for fam in families:
#     generate_graph(fam, "By Family", "pareto_decades")
    # generate_graph(fam, "By Family", "space")
    # generate_graph(fam, "By Family", "time")