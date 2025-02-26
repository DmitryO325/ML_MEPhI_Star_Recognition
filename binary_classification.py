import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, \
    recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# ОБРАБОТКА

# 1 - переменные звезды, 0 - статические

train = train.drop('Unnamed: 0', axis=1)
train = train[train['err'] < 1]  # удалил слишком большую ошибку

# нормализация (да, не влияет на лес, но как-то получше зашло)
mmsc = MinMaxScaler()
norm = mmsc.fit_transform(train)
train = pd.DataFrame(norm, columns=['fuv_mag', 'nuv_mag', 'Vmag', 'Bmag', 'err', 'variable'])

# ОБУЧЕНИЕ

X = train.drop(['variable'], axis=1)
Y = train['variable']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


model = RandomForestClassifier(
    class_weight={0: 1, 1: 30}, # усиленный вес для класса 1
    n_estimators=500, # большое количество деревьев
    max_depth=15, # большая максимальная глубина
    max_features='sqrt' # лучше для дисбаланса
)

model.fit(x_train, y_train)
answers_pred = model.predict(x_test)

print(accuracy_score(y_test,
                     answers_pred))  # ~0.97 количество правильных ответов ко всем ответам
print(confusion_matrix(y_test,
                       answers_pred))  # по столбцам: TP, FN (предсказан 0, на самом деле 1), FP, TP
print(recall_score(y_test, answers_pred))  # насколько хорошо находит 1
print(precision_score(y_test,
                      answers_pred))  # способность не ошибаться, когда говорит, что 1
print(f1_score(y_test,
               answers_pred))  # сочетание precision и recall, ~.39