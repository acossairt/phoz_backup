path_to_data_file = 'results.csv'

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pprint import pprint
from datetime import datetime
from tqdm import tqdm
from IPython.display import display, Markdown

width = 21
height = 13

matplotlib.rcParams.update({
    'font.size': 21,
    'figure.figsize': (width, height),
    'figure.facecolor': 'white',
    'savefig.dpi': 72,
    'figure.subplot.bottom': 0.125,
    'figure.edgecolor': 'white',
    'xtick.labelsize': 21,
    'ytick.labelsize': 21})

#######################

df = pd.read_csv(path_to_data_file)

display(Markdown(f'The search did _{df.count()[0]}_ **evaluations**.'))

print(df.head())

######################

print(df.describe())


#######################

plt.figure(543)
plt.plot(df.elapsed_sec, df.objective)
plt.ylabel('Objective')
plt.xlabel('Time (s.)')
plt.xlim(0)
plt.grid()
plt.savefig('./analytics_plots/objective.png')


#######################

plt.figure(333)
not_include = ['elapsed_sec']
sns.pairplot(df.loc[:, filter(lambda n: n not in not_include, df.columns)],
                diag_kind="kde", markers="+",
                plot_kws=dict(s=50, edgecolor="b", linewidth=1),
                diag_kws=dict(shade=True))


plt.savefig('./analytics_plots/pairplot.png')



#######################

plt.figure(332)
corr = df.loc[:, filter(lambda n: n not in not_include, df.columns)].corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap=sns.diverging_palette(220, 10, as_cmap=True))

plt.savefig('./analytics_plots/heatmap.png')


#######################
### BEST POINT
#######################


i_max = df.objective.idxmax()
best_point = dict(df.iloc[i_max])

print(best_point)

#########################
