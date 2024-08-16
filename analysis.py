import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

df = pd.read_csv('diabetes.csv')

#A summary of info as well as a look at first five rows within the dataset
print(df.info())
print(df.head())

#Plotting the histograms
df.hist()
plt.show()

#Create a subplot of 3x3
fig, axes = plt.subplots(3,3, figsize = (15,15))

#Plot a density plot for each variable
for idx, col in enumerate (df.columns[:-1]):
	ax = axes[idx // 3, idx % 3]
	ax.yaxis.set_ticklabels([])
	sns.kdeplot(df.loc[df.Outcome == 0][col], ax = ax, linestyle = '-', color = 'black', label = "No Diabetes")
	sns.kdeplot(df.loc[df.Outcome == 1][col], ax = ax, linestyle = '--', color = 'black', label = "Diabetes")
	ax.set_title(col)
	
#Hide the 9th subplot since there can only be 8 plots.
fig.delaxes(axes[2, 2])

plt.legend(loc='upper right')
plt.tight_layout()	
plt.show()

print(df.isnull().any())
print(df.describe())


