import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import data
my_df = pd.read_csv("production.csv")

# run correlation matrix and plot
f, ax = plt.subplots(figsize=(10, 8))
corr = my_df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=bool),
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
plt.show()