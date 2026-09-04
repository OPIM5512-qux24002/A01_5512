from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt

housing = fetch_california_housing(as_frame=True)
df = housing.frame
print(df.head())
print(df.shape)

plt.boxplot(df['MedHouseVal'])
plt.savefig('figs/boxplot.png')

