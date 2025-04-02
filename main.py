import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
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
# tsne = TSNE(random_state=42)
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

def count_metrics(y, y_pred):

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

# linear classification

# from sklearn.linear_model import LogisticRegression

X = df[[x for x in df.columns if x != 'present']]
y = df['present']

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# lr = LogisticRegression(max_iter=300, class_weight='balanced')
#
# # cross-validation
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# y_pred = cross_val_predict(lr, X_scaled, y, cv=cv)


# count_metrics(y, y_pred)

# # kNN, плохо подходит для несбалансированного датасета
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
#
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# y_pred = cross_val_predict(knn, X_scaled, y, cv=cv)
#
# count_metrics(y, y_pred)
#
# # decision tree
# from sklearn.tree import DecisionTreeClassifier
# tree = DecisionTreeClassifier(criterion="entropy", class_weight='balanced')
#
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# y_pred = cross_val_predict(tree, X_scaled, y, cv=cv)
#
# count_metrics(y, y_pred)

# random forest
# from sklearn.ensemble import RandomForestClassifier
#
# rfc = RandomForestClassifier(class_weight='balanced')
#
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# y_pred = cross_val_predict(rfc, X_scaled, y, cv=cv)
#
# count_metrics(y, y_pred)

# gradient boosting - recall 0.16
from sklearn.ensemble import GradientBoostingClassifier
#
# gb = GradientBoostingClassifier()
#
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# y_pred = cross_val_predict(gb, X_scaled, y, cv=cv)
#
# count_metrics(y, y_pred) # acc 0.91, prec 0.86, rec 0.16, f1 0.27

# подбор параметров
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

pipeline = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("gb", GradientBoostingClassifier())
])

# gb = GradientBoostingClassifier()
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# y_pred = cross_val_predict(pipeline, X_scaled, y, cv=cv)
#
# count_metrics(y, y_pred) # 0.86 acc, 0.4 prec, 0.72 rec, 0.52 F1

from sklearn.model_selection import GridSearchCV

params = {
    "smote__k_neighbors": [3, 5],
    "gb__max_depth": [7, 10, 12],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(pipeline, params, scoring="recall", cv=cv, n_jobs=-1)
grid_search.fit(X_scaled, y)

print("Лучшие параметры:", grid_search.best_params_)

best_model = grid_search.best_estimator_

y_pred = cross_val_predict(best_model, X_scaled, y, cv=cv)
count_metrics(y, y_pred)
