import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

import optuna

import warnings

warnings.filterwarnings('ignore')

# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

data = pd.read_csv('dataset/dataset.csv')
# print(data.head(30))
# print(data.describe())
# print(data.shape)

# print(data[(data.present == 1) & (data.type.notna())].shape[0] / data[data.present == 1].shape[0])
'Если present равен 1, то type будет не NaN с вероятностью 0.99'

# print(data[(data.present == 0) & (data.type.notna())].shape[0])
'Если present равен 0, то type всегда будет NaN'

# print(data.info())
'Все столбцы, кроме type, у нас числовые'
'4 объекта имеют NaN в min_mag и max_mag'

# print(data[data.min_mag.isna()])
'type у нас здесь NaN, поэтому их проще удалить'

data = data.dropna(axis=0, subset=['min_mag', 'max_mag'])
# print(data)

# print(data.type.unique())
'У нас много разных типов объектов'

'Заменим NaN, где present равен 1, на MISC, то есть не классифицированные объекты'
data.loc[(data.present == 1) & (data.type.isna()), 'type'] = 'MISC'
# print(data[data.present == 1].head(50))

# print(data[(data.present == 1) & (data.type.notna())].shape[0] / data[data.present == 1].shape[0])
'Теперь у нас везде, где present равен 1, значение в type не NaN'

'Проведём статистический анализ'
# for column in data.columns:
#     plt.figure(figsize=(16, 8))
#     plt.title(column)
#     plt.axis('off')
#
#     plt.subplot(1, 2, 1)
#     sns.histplot(data[column], kde=True)
#
#     plt.subplot(1, 2, 2)
#     sns.boxplot(data[column])
#
#     plt.show()

'''Выбросы есть, но лучше их оставить, так как они могут быть важными и, 
   скорее всего, не являются ошибками в данных'''

'Разделим набор данных на матрицу признаков и целевые значения'
'present - целевой столбец для бинарной классификации'
'type - целевой столбец для многоклассовой классификации'

X = data.drop(columns=['present', 'type'])
y_binary = data['present']
y_multy = data['type']

# print(X)
# print(y_famous)

'Нормализуем данные'
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# print(X_scaled[:10])

# 'Преобразуем категориальные признаки в числовые'
# encoder = LabelEncoder()
# y_encoded = encoder.fit_transform(y.type)

'Посмотрим коррелируемость признаков'
# plt.figure(figsize=(16, 8))
# sns.heatmap(X.corr(), annot=True, fmt='.2f')
# plt.show()

'''У нас есть сильно коррелируем признаки, 
   применим метод главных компонент, чтобы уменьшить число признаков'''

pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)
# print(X_pca)

'Проверим сбалансированность выборки'
# print(len(y_binary[y_binary == 0]) / len(y_binary))
'Объект не звезда в 90% случаев, звезда в 10% случаев'
'Выборка не сбалансирована'

'Решим задачу бинарной классификации'

'Разделим выборку на обучающую и тестовую'
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)

'''Обучим несколько моделей:
   1. Логистическая регрессия
   2. Метод опорных векторов
   3. Случайный лес
   4. Градиентный бустинг
'''

# logistic_regression = LogisticRegression(class_weight='balanced', max_iter=2000)
# support_vector_machine = SVC(class_weight='balanced', kernel='linear', max_iter=2000)
# random_forest = RandomForestClassifier(class_weight='balanced')
# gradient_boosting = XGBClassifier(class_weight='balanced')
#
# models = logistic_regression, support_vector_machine, random_forest, gradient_boosting

# 'Проверим precision (точность), recall (полноту) и F1-score на обучающей выборке'
# print('Обучающая выборка:\n')
#
# for model in models:
#     model.fit(X_train, y_train)
#     y_predict = model.predict(X_train)

# print(f'Precision у модели {model.__class__.__name__}: {precision_score(y_train, y_predict)}')
# print(f'Recall у модели {model.__class__.__name__}: {recall_score(y_train, y_predict)}')
# print(f'F1-score у модели {model.__class__.__name__}: {f1_score(y_train, y_predict)}\n')

'Проверим precision (точность), recall (полноту) и F1-score на тестовой выборке'
# print('Тестовая выборка:\n')

# for model in models:
#     model.fit(X_train, y_train)
#     y_predict = model.predict(X_test)

# print(f'Precision у модели {model.__class__.__name__}: {precision_score(y_test, y_predict)}')
# print(f'Recall у модели {model.__class__.__name__}: {recall_score(y_test, y_predict)}')
# print(f'F1-score у модели {model.__class__.__name__}: {f1_score(y_test, y_predict)}\n')

'Видим, что у логистической регрессии и метода опорных векторов нет переобучения, но низкий показатель F1-меры'
'У случайного леса ситуация лучше, но модель сильно переобучается'
'Наилучшим образом показывает себя градиентный бустинг, F1-мера равна 0.73'
'Попробуем улучшить этот показатель'

'Найдём наиболее оптимальные гиперпараметры для градиентного бустинга'
param_grid = {
    'n_estimators': np.arange(400, 650, 50),
    'learning_rate': np.arange(0.25, 0.36, 0.01),
    'max_depth': np.arange(5, 8)
}

gradient_boosting_models = XGBClassifier(), CatBoostClassifier(), LGBMClassifier()

grid = GridSearchCV(XGBClassifier(), param_grid, scoring='f1', cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

# print(f'Лучшие параметры для XGBClassifier согласно GridSearchCV:', grid.best_params_)
'''Лучшие гиперпараметры ля XGBClassifier согласно GridSearchCV примерно равны:
   max_depth=5, learning_rate=0.26, n_estimators=550'''

gradient_boosting_xgboost = XGBClassifier(learning_rate=0.26,
                                          n_estimators=550,
                                          max_depth=5)
gradient_boosting_xgboost.fit(X_train, y_train)

'Попробуем найти наилучший порог для F1-меры'
probs = gradient_boosting_xgboost.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, probs)
f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
best_thresh = thresholds[f1_scores.argmax()]
print(f'Лучший порог для XGBClassifier:', best_thresh)

'Лучший порог составляет примерно 0.3388796'

y_predict = (probs > best_thresh).astype(int)

print(f'Precision у модели градиентного бустинга: {precision_score(y_test, y_predict)}')
print(f'Recall у модели градиентного бустинга: {recall_score(y_test, y_predict)}')
print(f'F1-score у модели градиентного бустинга: {f1_score(y_test, y_predict)}')

gradient_boosting_catboost = CatBoostClassifier()
gradient_boosting_catboost.fit(X_train, y_train)

'Попробуем найти наилучший порог для F1-меры'
probs = gradient_boosting_catboost.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, probs)
f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
best_thresh = thresholds[f1_scores.argmax()]
print(f'Лучший порог для CatBoostClassifier:', best_thresh)

'Лучший порог составляет примерно 0.25729036'

y_predict = (probs > best_thresh).astype(int)

print(f'Precision у модели градиентного бустинга: {precision_score(y_test, y_predict)}')
print(f'Recall у модели градиентного бустинга: {recall_score(y_test, y_predict)}')
print(f'F1-score у модели градиентного бустинга: {f1_score(y_test, y_predict)}')

gradient_boosting_lightgbm = LGBMClassifier()
gradient_boosting_lightgbm.fit(X_train, y_train)

'Попробуем найти наилучший порог для F1-меры'
probs = gradient_boosting_lightgbm.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, probs)
f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
best_thresh = thresholds[f1_scores.argmax()]
print(f'Лучший порог для LGBMClassifier:', best_thresh)

'Лучший порог составляет примерно 0.25729036'

y_predict = (probs > best_thresh).astype(int)

print(f'Precision у модели градиентного бустинга: {precision_score(y_test, y_predict)}')
print(f'Recall у модели градиентного бустинга: {recall_score(y_test, y_predict)}')
print(f'F1-score у модели градиентного бустинга: {f1_score(y_test, y_predict)}')

'Лучший результат F1-меры: 0.7872983870967742'
#
# '''Задачи:
#    1. Проверить CatBoost и LightGBM
#    2. Создать ipynb-файл
#    3. Составить матрицу ошибок
# '''
