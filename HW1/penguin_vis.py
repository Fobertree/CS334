# %%
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from palmer import *
import os

# %%
df = pd.read_csv("./data/penguins.csv")

# %%
df.info()

# %%
df.describe()

# %%
len(df)

# %%
df['species'].value_counts()

# %%
ax = sns.histplot(df['species'])
ax.set(title="Species Count Histogram")

# %%
df2 = df.dropna(how = 'any', axis=0)

print(len(df2))
ax = sns.histplot(df2['species'])
ax.set(title="Species Count Histogram with Dropped NA Values")

# %%
df2

# %%
df2.isnull().values.any()

# %%
df2['species'].value_counts()

# %%
pd.DataFrame(to_numeric(df))

# %%
ax = plt.gca()
df.boxplot('bill_length_mm',ax=ax, by='species')
ax.get_figure().suptitle('Boxplot of penguins grouped by species')
ax.set(title="Bill Length (mm)", ylabel="Length (mm)", xlabel="Penguin Species")
ax.get_figure().savefig(os.path.join("data","Boxplot of penguins grouped by species, Bill Length (mm)"))

# %%
ax = plt.gca()
df.boxplot('bill_depth_mm', ax=ax, by='species')
ax.get_figure().suptitle('Boxplot of penguins grouped by species')
ax.set(title="Bill Depth (mm)", ylabel="Bill Depth (mm)", xlabel="Penguin Species")
ax.get_figure().savefig(os.path.join("data","Boxplot of penguins grouped by species, Bill Depth (mm)"))

# %%
ax = plt.gca()
df.boxplot('flipper_length_mm',ax=ax, by='species')
ax.get_figure().suptitle('Boxplot of penguins grouped by species')
ax.set(title="Flipper Length (mm)", ylabel="Length (mm)", xlabel="Penguin Species")
ax.get_figure().savefig(os.path.join("data","Boxplot of penguins grouped by species, Flipper Length (mm)"))


# %%
ax = plt.gca()
df.boxplot('body_mass_g', ax=ax, by='species')
ax.get_figure().suptitle('Boxplot of penguins grouped by species')
ax.set(title="Body Mass (g)", ylabel="Body Mass (g)", xlabel="Penguin Species")
ax.get_figure().savefig(os.path.join("data","Boxplot of penguins grouped by species, Body Mass (g)"))

# %%
#'bill_depth_mm', 'bill_length_mm', 'flipper_length_mm', 'body_mass_g'

# %%
# bill depth vs bill length
ax = sns.scatterplot(df, x='bill_depth_mm', y='bill_length_mm', hue='species')
ax.set(title='Bill Depth (mm) vs Bill Length (mm) Scatterplot by Species', xlabel="Bill Depth (mm)", ylabel = "Bill Length (mm)")
ax.get_figure().savefig(os.path.join("data","Bill Depth (mm) vs Bill Length (mm) Scatterplot by Species"))

# %%
df.columns

# %%
ax = sns.scatterplot(df, x='bill_depth_mm', y='flipper_length_mm', hue='species')
ax.set(title='Bill Depth (mm) vs Flipper Length (mm) Scatterplot by Species', xlabel="Bill Depth (mm)", ylabel = "Flipper Length (mm)")
ax.get_figure().savefig(os.path.join("data","Bill Depth (mm) vs Flipper Length (mm) Scatterplot by Species"))

# %%
ax = sns.scatterplot(df, x='bill_depth_mm', y='body_mass_g', hue='species')
ax.set(title='Bill Depth (mm) vs Body Mass (g) Scatterplot by Species', xlabel="Bill Depth (mm)", ylabel = "Body Mass (g)")
ax.get_figure().savefig(os.path.join("data","Bill Depth (mm) vs Body Mass (g) Scatterplot by Species"))

# %%
ax = sns.scatterplot(df, x='bill_length_mm', y='flipper_length_mm', hue='species')
ax.set(title='Bill Length (mm) vs Flipper Length (mm) Scatterplot by Species', xlabel="Bill Length (mm)", ylabel = "Flipper Length (mm)")
ax.get_figure().savefig(os.path.join("data","Bill Length (mm) vs Flipper Length (mm) Scatterplot by Species"))

# %%
ax = sns.scatterplot(df, x='bill_length_mm', y='body_mass_g', hue='species')
ax.set(title='Bill Length (mm) vs Body Mass(g) Scatterplot by Species', xlabel="Bill Length (mm)", ylabel = "Body Mass(g)")
ax.get_figure().savefig(os.path.join("data","Bill Length (mm) vs Body Mass(g) Scatterplot by Species"))

# %%
ax = sns.scatterplot(df, x='flipper_length_mm', y='body_mass_g', hue='species')
ax.set(title='Flipper Length (mm) vs Body Mass (g) Scatterplot by Species', xlabel="Flipper Length (mm)", ylabel = "Body Mass (g)")
ax.get_figure().savefig(os.path.join("data","Flipper Length (mm) vs Body Mass (g) Scatterplot by Species"))