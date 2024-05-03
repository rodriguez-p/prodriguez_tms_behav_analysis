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

# create a df selecting the necessary columns from the raw data
df = raw_data[['Subject', 'Awareness', 'Attention', 'Region', 'Task', 'Target.ACC', 'TargetPresence']].copy()

# create a new 'Awareness' column based on the 'Seen' column; we will use 'Seen' as the DV and 'Awareness' as the IV
df['Seen'] = df['Awareness']
awareness_mapping = {'1': 'Seen', '0': 'Unseen'}
df['Awareness'] = df['Seen'].map(awareness_mapping)

# rename the regions
region_mapping = {'F': 'FEF', 'V': 'Vertex'}
df['Region'] = df['Region'].replace(region_mapping)

# remove missing values and select only Present trials
df = df.dropna()
df = df[df["TargetPresence"] == "Present"]

## MAIN EFFECT REGION ACC PLOT
df_acc = df
mean_by_region = df_acc.groupby(['Region', 'Task', 'Subject'])['Target.ACC'].mean().reset_index()

dx = 'Region'
dy = 'Target.ACC'

fig, axes = plt.subplots()

sns.set(style = 'ticks', font_scale = 2, rc={"lines.linewidth": 0.7})

ax = sns.boxplot(data = mean_by_region, x = dx, y = dy, dodge=.8 - .8 / 3, whis = 0, showfliers = False)
sns.stripplot(data = mean_by_region, x = dx, y =dy,
              edgecolor = 'black', linewidth = 1, alpha = 0.4)
sns.pointplot(data = mean_by_region, x = dx, y = dy, estimator = np.median, 
              color = 'black', linestyles = '--')
pairs=[("FEF", "Vertex")]
annotator = Annotator(ax, pairs = pairs, data=mean_by_region, x=dx, y=dy)
annotator.set_custom_annotations(["*"]) 
annotator.configure(text_format='star', loc='outside')
annotator.annotate()
ax.set_ylabel('Accuracy')
sns.despine(trim=True)
plt.tight_layout()
plt.savefig('ACC_Region.png')
plt.close()

## MAIN EFFECT REGION SEEN PLOT
df_seen = df
df_seen["Seen"] = pd.to_numeric(df_seen["Seen"])

mean_by_region = df_seen.groupby(['Region', 'Task', 'Attention', 'Subject'])['Seen'].mean().reset_index()

dx = 'Region'
dy = 'Seen'

fig, axes = plt.subplots()

sns.set(style = 'ticks', font_scale = 2, rc={"lines.linewidth": 0.7})

ax = sns.boxplot(data = mean_by_region, x = dx, y = dy, dodge=.8 - .8 / 3, whis = 0, showfliers = False)
sns.stripplot(data = mean_by_region, x = dx, y =dy, 
              edgecolor = 'black', linewidth = 1, alpha = 0.4)
sns.pointplot(data = mean_by_region, x = dx, y = dy, estimator = np.median, 
              color = 'black', linestyles = '--')
pairs=[("FEF", "Vertex")]
annotator = Annotator(ax, pairs = pairs, data=mean_by_region, x=dx, y=dy)
annotator.set_custom_annotations(["*"])
annotator.configure(text_format='star', loc='outside')
annotator.annotate()
ax.set_ylabel('Proportion Seen')
sns.despine(trim=True)
plt.tight_layout()
plt.savefig('Seen_Region.png')
plt.close()