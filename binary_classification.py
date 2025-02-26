import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')


# ОБРАБОТКА

# 1 - переменные звезды, 0 - статические
train = train.drop('Unnamed: 0', axis=1)

# что-то сделать с выбросами
sns.boxplot(train['err'], width=0.3)
plt.show()

# корелляция невысокая, плохо
# sns.heatmap(train.corr(), annot = True, cmap="YlGnBu", linecolor='white',linewidths=1)
# plt.show()

# датасет сильно несбалансирован
# print(train['variable'].value_counts())

# ОБУЧЕНИЕ

X = train.drop(['variable'], axis=1)
Y = train['variable']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

# логистическая регрессия хуже из-за несбалансированных данных
# model = LogisticRegression()
model = RandomForestClassifier(class_weight={0: 1, 1: 15}) # штрафуем за ошибки класса 1 в 10 раз сильнее

model.fit(x_train, y_train)
answers_pred = model.predict(x_test)

# дисбаланс классов
print(accuracy_score(y_test, answers_pred)) # ~0.97 количество правильных ответов ко всем ответам
print(confusion_matrix(y_test, answers_pred)) # по столбцам: TP, FN (предсказан 0, на самом деле 1), FP, TP
print(recall_score(y_test, answers_pred)) # насколько хорошо находит 1
print(precision_score(y_test, answers_pred)) # способность не ошибаться, когда говорит, что 1
print(f1_score(y_test, answers_pred)) # сочетание precision и recall, ~.3 лес, ~.12 регрессия
