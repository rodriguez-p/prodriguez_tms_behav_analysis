from pymer4.models import Lmer
import pandas as pd
import janitor
import os
import numpy as np

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

## model for all (filtered) RT data
model = Lmer('Target.RT ~ Awareness*Attention*Region*Task + (1|Subject)', data = df)

model.fit(
    factors={"Awareness": ['Seen', 'Unseen'], 
             "Attention": ["Attended", "Unattended"],
             "Region": ["FEF", "Vertex"],
             "Task": ["Alerting", "Orienting"]
            },
    ordered=True,
    summarize=True,
    family = 'inverse_gaussian',
    conf_int = 'boot'
)

# generate ANOVA-like table
model_anova = model.anova()
print(model_anova)

# specify pairwise comparisons for the 'Task' factor
marginal_estimates, comparisons = model.post_hoc(
    marginal_vars=["Task"], p_adjust = "bonf"
)

print(comparisons)

# specify pairwise comparisons for the 'Attention' factor
marginal_estimates, comparisons = model.post_hoc(
    marginal_vars=["Attention"], p_adjust = "bonf"
)

print(comparisons)

## model for the seen trials 
df_aware = df[df["Awareness"] == "Seen"]

model_aware = Lmer('Target.RT ~ Attention*Region*Task + (1|Subject)', data = df_aware)

model_aware.fit(
    factors={"Awareness": ['Seen', 'Unseen'], 
             "Attention": ["Attended", "Unattended"],
             "Region": ["FEF", "Vertex"],
             "Task": ["Alerting", "Orienting"]
            },
    ordered=True,
    summarize=True,
    family = 'inverse_gaussian'
)

model_aware_anova = model_aware.anova()
print(model_aware_anova)

marginal_estimates, comparisons = model_aware.post_hoc(
    marginal_vars=["Attention", "Region"], p_adjust = "bonf"
)

print(comparisons)

marginal_estimates, comparisons = model_aware.post_hoc(
    marginal_vars=["Task", "Region"], p_adjust = "bonf"
)

print(comparisons)

# model for the unseen trials
df_unaware = df[df["Awareness"] == "Unseen"]

model_unaware = Lmer('Target.RT ~ Attention*Region*Task + (1|Subject)', data = df_unaware)

model_unaware.fit(
    factors={"Awareness": ['Seen', 'Unseen'], 
             "Attention": ["Attended", "Unattended"],
             "Region": ["FEF", "Vertex"],
             "Task": ["Alerting", "Orienting"]
            },
    ordered=True,
    summarize=True,
    family = 'inverse_gaussian'
)

model_unaware_anova = model_unaware.anova()
print(model_unaware_anova)

#%% model for the alerting trials (including seen and unseen)

df_alerting = df[df["Task"] == "Alerting"]

model_alerting = Lmer('Target.RT ~ Attention*Region + (1|Subject)', data = df_alerting)

model_alerting.fit(
    factors={"Awareness": ['Seen', 'Unseen'], 
             "Attention": ["Attended", "Unattended"],
             "Region": ["FEF", "Vertex"]
            },
    ordered=True,
    summarize=True,
    family = 'inverse_gaussian'
)

model_alerting_anova = model_alerting.anova()
print(model_alerting_anova)

#%% model for the orienting trials (including seen and unseen)

df_orienting = df[df["Task"] == "Orienting"]

model_orienting = Lmer('Target.RT ~ Attention*Region + (1|Subject)', data = df_orienting)

model_orienting.fit(
    factors={"Awareness": ['Seen', 'Unseen'], 
             "Attention": ["Attended", "Unattended"],
             "Region": ["FEF", "Vertex"]
            },
    ordered=True,
    summarize=True,
    family = 'inverse_gaussian'
)

model_orienting_anova = model_orienting.anova()
print(model_orienting_anova)
