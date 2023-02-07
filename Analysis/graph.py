import math
from os import truncate
from tkinter import E
from turtle import title
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import helpers


def generate_graph(family_name, variation, type):
    """
    type: one of ['time', 'space', 'both_time_improvements', 'both_tradeoffs', 'pareto_decades']
    """
    dataframe = pd.read_csv('Analysis/data.csv')
    dataframe = dataframe.replace(np.nan, '', regex=True)
    algorithms = dataframe.loc[dataframe['Family Name'] == family_name]
    algorithms = algorithms.loc[algorithms['Variation'] == variation]
    
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
        # plt.step(lower_x, lower_time, where='post', color="#FA9A20",
        #         linewidth=5, solid_capstyle='round')
        # plt.annotate('Trivial', xy=(lower_x[0], lower_time[0]), xytext=(
        #     lower_x[0], lower_time[0] + 0.15), color='#FA9A20', fontsize=12, weight='bold')

        # plt.step(upper_x, upper_time, where='post', color="#822600",
        #         linewidth=5, solid_capstyle='round')
        # plt.annotate('Brute Force', xy=(upper_x[0], upper_time[0]), xytext=(
        #     upper_x[0], upper_time[0] + 0.15), color='#822600', fontsize=12, weight='bold')

        # plt.plot(upper_x[1:-1], upper_time[1:-1], 'o', color='#C71F1D', markersize=14)
        # for i in range(1, len(upper_x) - 1):
        #     plt.annotate(upper_bound[i]['name'].split(';')[-1] + ', ' + str(upper_bound[i]['year']), xy=(upper_x[i], upper_time[i]),
        #                 xytext=(upper_x[i] + 1, upper_time[i] + 0.15), color='#C71F1D', fontsize=12, weight='bold')

        # plt.fill_between(upper_x, upper_time, [2] *
        #                  len(upper_time), step="post", color="#FFFAEB")
        # title_suffix = 'Time'

        save_dest = 'Analysis/Plots/Time Improvements/'

        # Get the algorithms that improve the time bounds
        improvements = []
        best_time = 8
        start_time = None
        improves = False
        algorithms = algorithms.sort_values('Year')
        for index, row in algorithms.iterrows():
            if not start_time:
                start_time = 8 - int(row['Starting Complexity'])
                best_time = start_time
                item = {'year': 1940, 'time': start_time, 'name': ''}
                improvements.append(item)
            item = {}
            item['year'] = int(row['Year'])
            item['time'] = float(row['Time Complexity Class']) - 1 # math.ceil()
            item['name'] = row['Algorithm Name']

            # See if this alg improves the time bound
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
        plt.plot(time_years[1:-1], time_class[1:-1], 'o', color='#C71F1D', markersize=14)
        for i in range(1, len(time_labels) - 1):
            plt.annotate(time_labels[i].split(';')[-1] + ', ' + str(time_years[i]), xy=(time_years[i], time_class[i]),
                        xytext=(time_years[i] + 1, time_class[i] + 0.15), color='#C71F1D', fontsize=12, weight='bold', rotation=45)
        title_suffix = 'Time'

    # Create space bound improvement step plots
    elif type == 'space':
        save_dest = 'Analysis/Plots/Space Improvements/'

        # Get the algorithms that improve the space bounds
        improvements = []
        best_space = 8
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
            if item['space'] < best_space:
                improvements.append(item)
                best_space = item['space']

        improves = len(improvements) > 1
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
        for i in range(0, len(space_labels) - 1):
            plt.annotate(space_labels[i].split(';')[-1] + ', ' + str(space_years[i]), xy=(space_years[i], space_class[i]),
                        xytext=(space_years[i] + 1, space_class[i] + 0.15), color='#C71F1D', fontsize=12, weight='bold', rotation=45)
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
        plt.xlabel('Space Complexity')
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
            item['space'] = item.get('space', 0)
            item['name'] = row['Algorithm Name']
            item['time'] = math.ceil(float(row['Time Complexity Class']) - 1)

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
                for time in range(0, item['time']):
                    if time in frontier:
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
            prev_decades = decades.keys()
            for time in frontier_algs:
                alg = frontier_algs[time][0]
                skip = False
                for x in prev_decades:
                    if decades[x][0]:
                        for prev_sp, prev_ti in zip(decades[x][0], decades[x][1]):
                            if prev_sp == alg['space'] and prev_ti == alg['time']:
                                skip = True
                                break
                        if skip:
                            break
                if not skip:
                    spaces.append(alg['space'])
                    times.append(alg['time'])
                    names.append(alg['name'])
                    years.append(alg['year'])

            decades[decade] = (spaces, times, names, years)

        # color = iter(cm.rainbow(np.linspace(0, 1, len(decades))))
        decades_list = [x for x in decades]
        decades_list.sort()
        points_plotted = 0
        for decade in decades_list[::-1]:
            label = str(decade) + '\'s'
            spaces, times, names, years = decades[decade]
            if spaces and times:
                # c = next(color)
                # plt.step(spaces, times, label=label, where='post', linewidth=5, c=c)
                points_plotted += len(names)
                plt.plot(spaces, times, 'o', markersize=14, label=label)
                for i in range(len(names)):
                    plt.annotate(names[i].split(';')[-1] + ', ' + str(years[i]), xy=(spaces[i], times[i]),
                                xytext=(spaces[i] + 0.15, times[i] + 0.15), color='#C71F1D', fontsize=12, weight='bold', rotation=0)#10)
        plt.legend(prop={'size': 24})
        plt.xlabel("Space Complexity", fontsize=20)
        plt.ylabel("Time Complexity", fontsize=20)
        if points_plotted > 1:
            save_dest = "Analysis/Plots/Pareto Decades/Improvements/"
        else:
            save_dest = "Analysis/Plots/Pareto Decades/No Improvements/"
        # save_dest = "Analysis/Plots/Pareto Decades/"
        title_suffix = "Pareto Frontier"



    plt.box(on=None)
    plt.tick_params(axis='x', colors='#A6A6A6')
    plt.tick_params(axis='y', colors='#A6A6A6')

    if(variation):
        plt.title(variation + ' ('+family_name+')',
                  fontsize=20, color='#A6A6A6')
    else:
        plt.title(family_name, fontsize=20, color='#A6A6A6')
    plt.tick_params(axis='x',   which='both',   bottom=False,
                    top=False, labelbottom=True)

    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7], ('constant', 'logn', 'linear',
                                          'nlogn', 'quadratic', 'cubic', 'poly > cubic', 'exponential'), weight='bold', fontsize=14)
    if type == 'both_tradeoffs' or type == 'pareto_decades':
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], ('constant', 'logn', 'linear',
                                          'nlogn', 'quadratic', 'cubic', 'poly > cubic', 'exponential'), weight='bold', fontsize=14)
        plt.grid(axis='both', color='#E8E8E8')
    else:
        plt.xticks(weight='bold', fontsize=14)
        plt.grid(axis='y', color='#E8E8E8')
    plt.tight_layout()

    try:
        plt.savefig(save_dest+family_name+' - '+variation+' - '+title_suffix+ ' Bounds Chart.png', dpi=300, bbox_inches='tight')
    except Exception as e:
        print(e)
        print("Failed")
    plt.clf()
    plt.close("all")

# family = 'Maximum Subarray Problem'
# variation = '1D Maximum Subarray'
# family = 'Optimal Binary Search Trees'
# variation = 'OBST'
# family = 'Sorting'
# variation = 'Non-Comparison Sorting'
# family = 'Sorting'
# variation = 'Comparison Sorting'

# generate_graph(family, variation, 'space')
# generate_graph(family, variation, 'time')
# generate_graph(family, variation, 'both_time_improvements')
# generate_graph(family, variation, 'both_tradeoffs')
# generate_graph(family, variation, 'pareto_decades')

helpers.clean_data()
families = helpers.get_families()
for fam in families:
    for var in helpers.get_variations(fam):
        try:
            generate_graph(fam, var, 'pareto_decades')
            generate_graph(fam, var, 'space')
            generate_graph(fam, var, 'time')
        except Exception as e:
            print(fam, var, e)