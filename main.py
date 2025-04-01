import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import RobustScaler

df = pd.read_csv('./data/whole_data_practice3.csv')
pd.set_option('display.max_columns', None)
plt.rcParams['figure.figsize'] = (16, 12)

# print(df['present'].value_counts()) # dataset is not balansed
# print(df['type'].isnull().sum() / df.shape[0]) # only 10% is filled
df.drop('type', inplace=True, axis=1)

# print(df.isnull().sum())
df.drop(df.index[:5], inplace=True)

# # some fancy graphics
#
# # correlation
# plt.figure()
# corr_matrix = df.corr()
# sns.heatmap(corr_matrix, annot=True)
# plt.show() # we can drop Bmag, gpmag, rpmag, ipmag since they correlate with Vmag
#
# df.drop(['Bmag', 'gpmag', 'rpmag', 'ipmag'], inplace=True, axis=1)
#
# # histogram
# df.drop(['present'], axis=1).hist()
# plt.tight_layout()
# plt.show()
#
# plt.figure()
# plt.boxplot(df['nobs'])
# plt.show() # majority is < 30, but some values are very big
#
# plt.figure()
# plt.violinplot(df[df['nobs'] < 30]['nobs'])
# plt.show() # most of the values are close to zero and less than 10
#
# plt.figure()
# plt.violinplot(df['RAJ2000'])
# plt.show() # interesting distribution

# the last one, I swear

# plt.figure()
# tsne = TSNE(random_state=17)
# X_repr = tsne.fit_transform(df)
# plt.scatter(X_repr[df['present']==1, 0],
#             X_repr[df['present']==1, 1], alpha=.2, c='blue',
#            label='constant')
# plt.scatter(X_repr[df['present']==0, 0],
#             X_repr[df['present']==0, 1], alpha=.2, c='orange',
#            label='pulsar')
# plt.legend()
# plt.show()

# let's go

# linear classification

# TODO: подобрать параметры
X = df[[x for x in df.columns if x != 'present']]
y = df['present']

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=300, class_weight='balanced')

# cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(model, X_scaled, y, cv=cv)


accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)


print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')

# confusion matrix
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()