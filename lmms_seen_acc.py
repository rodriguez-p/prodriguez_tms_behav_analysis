from pymer4.models import Lmer
import pandas as pd
import os
import seaborn as sns
import matplotlib
import dataframe_image as dfi
import matplotlib.pyplot as plt
import numpy as np
from statannotations.Annotator import Annotator
import dataframe_image as dfi

# set pandas to show 3 decimals
pd.options.display.float_format = '{:,.3f}'.format
pd.set_option("display.precision", 3)

# read the csv file
raw_data = pd.read_csv('raw_data_all.csv', sep = ';')

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

#%% model for ACC analysis
df_acc = df
model_acc = Lmer('Target.ACC ~ Awareness*Attention*Region*Task + (1|Subject)', data = df_acc)

model_acc.fit(
    factors={"Awareness": ['Seen', 'Unseen'], 
             "Attention": ["Attended", "Unattended"],
             "Region": ["FEF", "Vertex"],
             "Task": ["Alerting", "Orienting"]
            },
    ordered=True,
    summarize=True,
    family = 'binomial'
)

model_acc_anova = model_acc.anova()
print(model_acc_anova)

marginal_estimates, comparisons = model_acc.post_hoc(
    marginal_vars=["Region"], p_adjust = "bonf")

print(comparisons)

#%% model for Seen analysis
df_seen = df
df_seen["Seen"] = pd.to_numeric(df_seen["Seen"])

model_seen = Lmer('Seen ~ Attention*Region*Task + (1|Subject)', data = df)

model_seen.fit(
    factors={"Attention": ["Attended", "Unattended"],
             "Region": ["FEF", "Vertex"],
             "Task": ["Alerting", "Orienting"]
            },
    ordered=True,
    summarize=True,
    family = 'binomial'
)

model_seen_anova = model_seen.anova()
print(model_seen_anova)

marginal_estimates, comparisons = model_seen.post_hoc(
    marginal_vars=["Task"], p_adjust = "bonf")

print(comparisons)

marginal_estimates, comparisons = model_seen.post_hoc(
    marginal_vars=["Attention"], p_adjust = "bonf")

print(comparisons)

#%%
media_por_region = df_seen.groupby(['Region', 'Task', 'Attention', 'Subject'])['Seen'].mean().reset_index()

dx = 'Region'
dy = 'Seen'
df_seen = media_por_region

fig, axes = plt.subplots()

sns.set(style = 'ticks', font_scale = 2, rc={"lines.linewidth": 0.7})

ax = sns.boxplot(data = df_seen, x = dx, y = dy, dodge=.8 - .8 / 3, whis = 0, showfliers = False)
sns.stripplot(data = df_seen, x = dx, y =dy, 
              edgecolor = 'black', linewidth = 1, alpha = 0.4)
sns.pointplot(data = df_seen, x = dx, y = dy, estimator = np.median, 
              color = 'black', linestyles = '--')
# sns.lineplot(x=dx, y=dy, data = df_seen, hue='Subject', marker='o',
#                   palette=['gray'] * len(df['Subject'].unique()),
#                   legend=False, alpha = 0.3, ci = None)
pairs=[("FEF", "Vertex")]
annotator = Annotator(ax, pairs = pairs, data=df_seen, x=dx, y=dy)
annotator.set_custom_annotations(["*"]) # ACTUALIZAR SEGÚN EL RESULTADO DEL TEST
annotator.configure(text_format='star', loc='outside')
annotator.annotate()
ax.set_ylabel('Proportion Seen')
sns.despine(trim=True)
plt.tight_layout()
plt.savefig('Seen_Region.png')
plt.close()
