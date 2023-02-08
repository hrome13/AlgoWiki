import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns
from tabulate import tabulate
from scipy import stats

def hist_lower_bound_powers():
    data = pd.read_csv("Reductions/reductions_data.csv")
    powers = data[data["Implied lower bound power"] != np.nan]["Implied lower bound power"]
    powers = pd.to_numeric(powers, errors="coerce")
    powers = powers.replace(np.nan, "Variable")
    powers = powers.astype(str)
    sns.set_theme()
    counts = powers.value_counts()
    counts = [counts.loc["Variable"], counts.loc["0.5"], counts.loc["1.0"], counts.loc["1.5"], counts.loc["2.0"], counts.loc["3.0"]]
    plt.bar(range(len(counts)), counts)
    plt.xticks(range(len(counts)), ["Variable", "0.5", "1", "1.5", "2", "3"])
    plt.title("Implied Lower Bound Powers")
    plt.ylabel("Number of Reductions")
    plt.savefig("Reductions/Plots/Implied Lower Bound Power Histogram")

hist_lower_bound_powers()