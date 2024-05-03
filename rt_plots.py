import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import janitor
import os
import seaborn as sns
import numpy as np
from statannotations.Annotator import Annotator

# set pandas to show 3 decimals
pd.options.display.float_format = '{:,.3f}'.format
pd.set_option("display.precision", 3)

# read the csv file (already filtered rt data)
# filter: >150ms, +/- 2.5 SD
raw_data = pd.read_csv('rt_filtered.csv', sep = ';')

# create a df selecting the necessary columns from the raw data
df = raw_data[['Subject', 'Seen', 'Attention', 'Region', 'Task', 'Target.RT', 'TargetPresence']].copy()

# create a new 'Awareness' column based on the 'Seen' column; we will use 'Seen' as the DV and 'Awareness' as the IV
awareness_mapping = {'1': 'Seen', '0': 'Unseen'}
df['Awareness'] = df['Seen'].map(awareness_mapping)

# rename the regions
region_mapping = {'F': 'FEF', 'V': 'Vertex'}
df['Region'] = df['Region'].replace(region_mapping)

# remove missing values and select only Present trials
df = df.dropna()
df = df[df["TargetPresence"] == "Present"]

## MAIN EFFECT REGION PLOT
# computes mean for all subjects and generates the plots

mean_by_region = df.groupby(['Region', 'Task', 'Subject'])['Target.RT'].mean().reset_index()
print(mean_by_region)

fig, axes = plt.subplots()

sns.set(style = 'ticks', font_scale = 2, rc={"lines.linewidth": 0.7})

ax = sns.boxplot(data = mean_by_region, x = 'Region', y = 'Target.RT', dodge=.8 - .8 / 3, whis = 0, showfliers = False)
sns.stripplot(data = mean_by_region, x = 'Region', y ='Target.RT', 
              edgecolor = 'black', linewidth = 1, alpha = 0.4)
sns.pointplot(data = mean_by_region, x = 'Region', y = 'Target.RT', estimator = np.meann, 
              color = 'black', linestyles = '--')
# annotator just used for visualization
pairs=[("FEF", "Vertex")]
annotator = Annotator(ax, pairs = pairs, data=mean_by_region, x=dx, y=dy)
annotator.set_custom_annotations(["*"]) # update acording to test results
annotator.configure(text_format='star', loc='outside')
annotator.annotate()
ax.set_ylabel('Reaction times (ms)')
sns.despine(trim=True)
plt.tight_layout()
plt.savefig('RT_Region.png')
plt.close()

## TASK * REGION PLOT
mean_by_region = df.groupby(['Region', 'Task', 'Subject'])['Target.RT'].mean().reset_index()
mean_alerting = mean_by_region[mean_by_region['Task'] == 'Alerting']
mean_orienting = mean_by_region[mean_by_region['Task'] == 'Orienting']
print(mean_by_region)

mean_alerting = mean_by_region[mean_by_region['Task'] == 'Alerting']
mean_orienting = mean_by_region[mean_by_region['Task'] == 'Orienting']

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

sns.set(style = 'ticks', font_scale = 2, rc={"lines.linewidth": 0.7})

sns.boxplot(data=mean_alerting, x='Region', y='Target.RT', dodge=.8 - .8 / 3, ax=axes[0], whis = 0, showfliers = False)
stripplot1 = sns.stripplot(data=mean_alerting, x='Region', y='Target.RT', 
              ax=axes[0], edgecolor = 'black', linewidth = 1, alpha = 0.4)
sns.pointplot(data = mean_alerting, x = 'Region', y = 'Target.RT', estimator = np.meann, 
              color = 'black', linestyles = '--', ax = axes[0])
sns.despine(trim=True)

pairs=[("Alerting", "Orienting")]
annotator = Annotator(ax = axes[0], pairs = pairs, data=mean_by_region, x=dx, y=dy)
annotator.set_custom_annotations(["*"]) 
annotator.configure(text_format='star', loc='outside')
annotator.annotate()

axes[0].set_xlabel('Alerting')
axes[0].set_ylabel('Reaction times (ms)')     

sns.boxplot(data=mean_orienting, x='Region', y='Target.RT', ax=axes[1], dodge=.8 - .8 / 3, whis = 0, showfliers = False)
stripplot2 = sns.stripplot(data=mean_orienting, x='Region', y='Target.RT', 
              ax=axes[1], edgecolor = 'black', linewidth = 1, alpha = 0.4)
sns.pointplot(data = mean_orienting, x = 'Region', y = 'Target.RT', estimator = np.meann, 
              color = 'black', linestyles = '--', ax = axes[1])
sns.despine(trim=True)

pairs=[("Alerting", "Orienting")]
annotator = Annotator(ax = axes[1], pairs = pairs, data=mean_by_region, x=dx, y=dy)
annotator.set_custom_annotations(["*"])
annotator.configure(text_format='star', loc='outside')
annotator.annotate()

axes[1].set_xlabel('Orienting')
axes[1].set_ylabel(' ')
axes[1].axes.get_yaxis().set_visible(False)
axes[1].spines['left'].set_visible(False)

plt.tight_layout()
plt.savefig('RT_TaskxRegion.png')
plt.close()

#%% ATTENTION * REGION PLOT
mean_by_region = df.groupby(['Region', 'Attention', 'Subject'])['Target.RT'].mean().reset_index()
mean_fef = mean_by_region[mean_by_region['Region'] == 'FEF']
mean_vertex = mean_by_region[mean_by_region['Region'] == 'Vertex']

dx = 'Attention'
dy = 'Target.RT'
dhue = 'Region'

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

sns.set(style = 'ticks', rc={"lines.linewidth": 0.7})

sns.boxplot(data=mean_fef, x = dx, y = dy, dodge=.8 - .8 / 3, ax=axes[0], whis = 0, showfliers = False, palette = 'Set3')
stripplot1 = sns.stripplot(data=mean_fef, x = dx, y = dy, dodge=False, 
              ax=axes[0], edgecolor = 'black', linewidth = 1, alpha = 0.4, palette = 'Set3')
sns.pointplot(data = mean_fef, x = dx, y = dy, estimator = np.meann, 
              color = 'black', linestyles = '--', ax = axes[0])
sns.despine(trim=True)

pairs=[("Attended", "Unattended")]
annotator = Annotator(ax = axes[0], pairs = pairs, data=mean_fef, x=dx, y=dy)
annotator.set_custom_annotations(["*"]) # ACTUALIZAR SEGÚN EL RESULTADO DEL TEST
annotator.configure(text_format='star', loc='outside')
annotator.annotate()

# axes[0].set_title('FEF')
axes[0].set_xlabel('FEF')
axes[0].set_ylabel('Target.RT')

sns.boxplot(data=mean_vertex, x = dx, y = dy, ax=axes[1], dodge=.8 - .8 / 3, whis = 0, showfliers = False, palette = 'Set3')
stripplot2 = sns.stripplot(data=mean_vertex, x = dx, y = dy, dodge=False, 
              ax=axes[1], edgecolor = 'black', linewidth = 1, alpha = 0.4, palette = 'Set3')
sns.pointplot(data = mean_vertex, x = dx, y = dy, estimator = np.meann, 
              color = 'black', linestyles = '--', ax = axes[1])
sns.despine(trim=True)

pairs=[("Attended", "Unattended")]
annotator = Annotator(ax = axes[1], pairs = pairs, data=mean_fef, x=dx, y=dy)
annotator.set_custom_annotations(["*"]) # ACTUALIZAR SEGÚN EL RESULTADO DEL TEST
annotator.configure(text_format='star', loc='outside')
annotator.annotate()

# axes[1].set_title('Vertex')
axes[1].set_xlabel('Vertex')
axes[1].set_ylabel(' ')
axes[1].axes.get_yaxis().set_visible(False)
axes[1].spines['left'].set_visible(False)

plt.tight_layout()
plt.show()

#%% ATTENTION X REGION AWARE
df_aware = df[df["Awareness"] == "Seen"]

mean_by_region = df_aware.groupby(['Region', 'Attention', 'Subject'])['Target.RT'].mean().reset_index()
mean_fef = mean_by_region[mean_by_region['Region'] == 'FEF']
mean_vertex = mean_by_region[mean_by_region['Region'] == 'Vertex']

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

sns.set(style = 'ticks', font_scale = 2, rc={"lines.linewidth": 0.7})

sns.boxplot(data=mean_fef, x = 'Attention', y = 'Target.RT', dodge=.8 - .8 / 3, ax=axes[0], whis = 0, showfliers = False, palette = 'Set3')
stripplot1 = sns.stripplot(data=mean_fef, x = 'Attention', y = 'Target.RT', 
              ax=axes[0], edgecolor = 'black', linewidth = 1, alpha = 0.4, palette = 'Set3')
sns.pointplot(data = mean_fef, x = 'Attention', y = 'Target.RT', estimator = np.meann, 
              color = 'black', linestyles = '--', ax = axes[0])
sns.despine(trim=True)

pairs=[("Attended", "Unattended")]
annotator = Annotator(ax = axes[0], pairs = pairs, data=mean_fef, x=dx, y=dy)
annotator.set_custom_annotations(["*"]) 
annotator.configure(text_format='star', loc='outside')
annotator.annotate()

axes[0].set_xlabel('FEF')
axes[0].set_ylabel('Reaction times (ms)')

sns.boxplot(data=mean_vertex, x = dx, y = dy, ax=axes[1], dodge=.8 - .8 / 3, whis = 0, showfliers = False, palette = 'Set3')
stripplot2 = sns.stripplot(data=mean_vertex, x = dx, y = dy, 
              ax=axes[1], edgecolor = 'black', linewidth = 1, alpha = 0.4, palette = 'Set3')
sns.pointplot(data = mean_vertex, x = dx, y = dy, estimator = np.meann, 
              color = 'black', linestyles = '--', ax = axes[1])
sns.despine(trim=True)

pairs=[("Attended", "Unattended")]
annotator = Annotator(ax = axes[1], pairs = pairs, data=mean_fef, x=dx, y=dy)
annotator.set_custom_annotations(["*"])
annotator.configure(text_format='star', loc='outside')
annotator.annotate()

axes[1].set_xlabel('Vertex')
axes[1].set_ylabel(' ')
axes[1].axes.get_yaxis().set_visible(False)
axes[1].spines['left'].set_visible(False)

plt.tight_layout()
plt.savefig('RT_AttentionxRegion_aware.png')
plt.close()

#%% REGION X TASK AWARE
mean_by_region = df_aware.groupby(['Region', 'Task', 'Subject'])['Target.RT'].mean().reset_index()
mean_alerting = mean_by_region[mean_by_region['Task'] == 'Alerting']
mean_orienting = mean_by_region[mean_by_region['Task'] == 'Orienting']
print(mean_by_region)

mean_alerting = mean_by_region[mean_by_region['Task'] == 'Alerting']
mean_orienting = mean_by_region[mean_by_region['Task'] == 'Orienting']

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

sns.set(style = 'ticks',  font_scale = 2, rc={"lines.linewidth": 0.7})

sns.boxplot(data=mean_alerting, x='Region', y='Target.RT', dodge=.8 - .8 / 3, ax=axes[0], whis = 0, showfliers = False)
stripplot1 = sns.stripplot(data=mean_alerting, x='Region', y='Target.RT', 
              ax=axes[0], edgecolor = 'black', linewidth = 1, alpha = 0.4)
sns.pointplot(data = mean_alerting, x = 'Region', y = 'Target.RT', estimator = np.meann, 
              color = 'black', linestyles = '--', ax = axes[0])
sns.despine(trim=True)

pairs=[("Alerting", "Orienting")]
annotator = Annotator(ax = axes[0], pairs = pairs, data=mean_by_region, x=dx, y=dy)
annotator.set_custom_annotations(["*"])
annotator.configure(text_format='star', loc='outside')
annotator.annotate()

axes[0].set_xlabel('Alerting')
axes[0].set_ylabel('Reaction times (ms)')     

sns.boxplot(data=mean_orienting, x='Region', y='Target.RT', ax=axes[1], dodge=.8 - .8 / 3, whis = 0, showfliers = False)
stripplot2 = sns.stripplot(data=mean_orienting, x='Region', y='Target.RT',
              ax=axes[1], edgecolor = 'black', linewidth = 1, alpha = 0.4)
sns.pointplot(data = mean_orienting, x = 'Region', y = 'Target.RT', estimator = np.meann, 
              color = 'black', linestyles = '--', ax = axes[1])
# sns.lineplot(x='Region', y='Target.RT', data=mean_orienting, hue='Subject', marker='o',
#                   palette=['gray'] * len(df['Subject'].unique()),
#                   legend=False, alpha = 0.3, ci = None, ax = axes[1])
sns.despine(trim=True)

axes[1].set_xlabel('Orienting')
axes[1].set_ylabel(' ')
axes[1].axes.get_yaxis().set_visible(False)
axes[1].spines['left'].set_visible(False)

plt.tight_layout()
plt.savefig('RT_TaskxRegion_aware.png')
plt.close()


