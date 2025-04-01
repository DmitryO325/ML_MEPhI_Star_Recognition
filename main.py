import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.manifold import TSNE

df = pd.read_csv('./data/whole_data_practice3.csv')
pd.set_option('display.max_columns', None)
plt.rcParams['figure.figsize'] = (16, 12)

print(df['present'].value_counts()) # dataset is not balansed
print(df['type'].isnull().sum() / df.shape[0]) # only 10% is filled
df.drop('type', inplace=True, axis=1)

print(df.isnull().sum())

# some fancy graphics

# correlation
plt.figure()
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show() # we can drop Bmag, gpmag, rpmag, ipmag since they correlate with Vmag

df.drop(['Bmag', 'gpmag', 'rpmag', 'ipmag'], inplace=True, axis=1)

# histogram
df.drop(['present'], axis=1).hist()
plt.tight_layout()
plt.show()

plt.figure()
plt.boxplot(df['nobs'])
plt.show() # majority is < 30, but some values are very big

plt.figure()
plt.violinplot(df[df['nobs'] < 30]['nobs'])
plt.show() # most of the values are close to zero and less than 10

plt.figure()
plt.violinplot(df['RAJ2000'])
plt.show() # interesting distribution

# the last one, I swear

# plt.figure()
# tsne = TSNE(random_state=17)
# tmp = df.copy()
# tmp.drop(df.index[:5], inplace=True)
# X_repr = tsne.fit_transform(tmp)
# plt.scatter(X_repr[tmp['present']==1, 0],
#             X_repr[tmp['present']==1, 1], alpha=.2, c='blue',
#            label='constant')
# plt.scatter(X_repr[tmp['present']==0, 0],
#             X_repr[tmp['present']==0, 1], alpha=.2, c='orange',
#            label='pulsar')
# plt.legend()
# plt.show()

# let's go

X = df[[x for x in df.columns if x != 'present']]
y = df['present']

# linear classification
