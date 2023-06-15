# AlgoWiki

In this repository, within the Analysis/ folder, all of my analysis of algorithms' space/time complexity data is carried out. There are some helper functions to clean the data in Analysis/helpers.py, and most of the actual code for the plots in Analysis/Plots/ is in Analysis/space_analysis.py.

Workflow:
- Download the Algorithm Wiki data as a csv Analysis/data_dirty.csv
- Before creating plots, run the function helpers.clean_data(), which creates a new csv Analysis/data_clean.csv
- While creating plots, the actual numbers are printed in the console
- If desired, use those numbers to create nicer plots in Excel


Additionally, within the Algorithms/ folder, there are folders representing "Problem Families". Within these are Python3 implementations of algorithms solving the problems in Jupyter Notebooks.

Currently included problem families:
- Line Simplification
