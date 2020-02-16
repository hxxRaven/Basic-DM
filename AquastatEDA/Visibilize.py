'''
评估目标和变量之间的关系，二维画图

二维绘图

CATEGORICAL X CATEGORICAL
Heat map of contingency table（http://seaborn.pydata.org/generated/seaborn.heatmap.html?highlight=heatmap#seaborn.heatmap）
Multiple bar plots（http://seaborn.pydata.org/tutorial/categorical.html?highlight=bar%20plot#bar-plots）

CATEGORICAL X CONTINUOUS
Box plots of continuous for each category（http://seaborn.pydata.org/generated/seaborn.boxplot.html#seaborn.boxplot）
Violin plots of continuous distribution for each category（http://seaborn.pydata.org/examples/simple_violinplots.html）
Overlaid histograms (if 3 or less categories)（http://seaborn.pydata.org/tutorial/distributions.html#histograms）

CONTINUOUS X CONTINOUS
Scatter plots（http://seaborn.pydata.org/examples/marginal_ticks.html?highlight=scatter）
Hexibin plots（http://seaborn.pydata.org/tutorial/distributions.html#hexbin-plots）
Joint kernel density estimation plots（http://seaborn.pydata.org/tutorial/distributions.html#kernel-density-estimation）
Correlation matrix heatmap（http://seaborn.pydata.org/examples/network_correlations.html?highlight=correlation）
'''

# plotting
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context("poster", font_scale=1.3)
import folium

# system packages
import os, sys
import warnings
warnings.filterwarnings('ignore')

# basic wrangling
import numpy as np
import pandas as pd

# eda tools
import pivottablejs
import missingno as msno
import pandas_profiling

# interactive
import ipywidgets as widgets

# more technical eda
import sklearn
import scipy

from aqua_helper import *

data = pd.read_csv('data/aquastat.csv.gzip', compression='gzip')

data.region = data.region.apply(lambda x: simple_regions[x])

data = data.loc[~data.variable.str.contains('exploitable'),:]
data = data.loc[~(data.variable=='national_rainfall_index')]

recent = time_slice(data, '2013-2017')

#GDP和季节的关系（两个连续值，散点图）
plt.scatter(recent['seasonal_variability'], recent['gdp_per_capita'])
plt.xlabel('Seasonal variability')
plt.ylabel('GDP per capita ($USD/person)')

##也可用sns画出两者之间和两者自身变化
svr = [recent.seasonal_variability.min(), recent.seasonal_variability.max()]
gdpr = [(recent.gdp_per_capita.min()), recent.gdp_per_capita.max()]
gdpbins = np.logspace(*np.log10(gdpr), 25)
g =sns.JointGrid(x="seasonal_variability", y="gdp_per_capita", data=recent, ylim=gdpr)
g.ax_marg_x.hist(recent.seasonal_variability, range=svr)
g.ax_marg_y.hist(recent.gdp_per_capita, range=gdpr, bins=gdpbins, orientation="horizontal")
g.plot_joint(plt.hexbin, gridsize=25)
ax = g.ax_joint
# ax.set_yscale('log')
g.fig.set_figheight(8)
g.fig.set_figwidth(9)


#相关度量两个变量之间的*线性关系的强度。我们可以使用相关性来识别变量（下面为两两计算相关系数corr()）
recent_corr = recent.corr().loc['gdp_per_capita'].drop(['gdp','gdp_per_capita'])
print(recent.corr())
def conditional_bar(series, bar_colors=None, color_labels=None, figsize=(13,24),
                   xlabel=None, by=None, ylabel=None, title=None):
    fig, ax  = plt.subplots(figsize=figsize)
    if not bar_colors:
        bar_colors = mpl.rcParams['axes.prop_cycle'].by_key()['color'][0]
    plt.barh(range(len(series)),series.values, color=bar_colors)
    plt.xlabel('' if not xlabel else xlabel);
    plt.ylabel('' if not ylabel else ylabel)
    plt.yticks(range(len(series)), series.index.tolist())
    plt.title('' if not title else title);
    plt.ylim([-1,len(series)]);
    if color_labels:
        for col, lab in color_labels.items():
            plt.plot([], linestyle='',marker='s',c=col, label= lab);
        lines, labels = ax.get_legend_handles_labels();
        ax.legend(lines[-len(color_labels.keys()):], labels[-len(color_labels.keys()):], loc='upper right');
    plt.close()
    return fig

#参数设置
bar_colors = ['#0055A7' if x else '#2C3E4F' for x in list(recent_corr.values < 0)]
color_labels = {'#0055A7':'Negative correlation', '#2C3E4F':'Positive correlation'}

conditional_bar(recent_corr.apply(np.abs), bar_colors, color_labels,
               title='Magnitude of correlation with GDP per capita, 2013-2017',
               xlabel='|Correlation|')


#连续值映射分类bins
plot_hist(recent, 'gdp_per_capita', xlabel='GDP per capita ($)', logx=True,
         ylabel='Number of countries', bins=25,
          title='Distribution of log GDP per capita, 2013-2017')
#分成5个大块，25个小块
capita_bins = ['Very low', 'Low', 'Medium', 'High', 'Very high']
recent['gdp_bin'] = pd.qcut(recent.gdp_per_capita, 5, capita_bins)
bin_ranges = pd.qcut(recent.gdp_per_capita, 5).unique()
#脚本
def plot_hist(df, variable, bins=None, xlabel=None, by=None,
              ylabel=None, title=None, logx=False, ax=None):
    if not ax:
        fig, ax = plt.subplots(figsize=(12, 8))
    if logx:
        bins = np.logspace(np.log10(df[variable].min()),
                           np.log10(df[variable].max()), bins)
        ax.set_xscale("log")

    if by:
        if type(df[by].unique()) == pd.Categorical:
            cats = df[by].unique().categories.tolist()
        else:
            cats = df[by].unique().tolist()

        for cat in cats:
            to_plot = df[df[by] == cat][variable].dropna()
            ax.hist(to_plot, bins=bins)
    else:
        ax.hist(df[variable].dropna().values, bins=bins)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    return ax

plot_hist(recent, 'gdp_per_capita', xlabel='GDP per capita ($)', logx=True,
         ylabel='Number of countries', bins=25, by='gdp_bin',
          title='Distribution of log GDP per capita, 2013-2017')
#同样的数据乐意用box图表示，从下到上，0-1/4-2/4-3/4-1位数
recent[['gdp_bin','total_pop_access_drinking']].boxplot(by='gdp_bin')
# plt.ylim([0,100000])
plt.title('Distribution of percent of total population with access to drinking water across gdp per capita categories');
plt.xlabel('GDP per capita quintile')
plt.ylabel('Total population of country')