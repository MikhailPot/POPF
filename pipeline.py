
# Импорт Библиотек
import pandas as pd
import numpy as np
import seaborn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn. linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from warnings import simplefilter
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from numpy import nan
import shap 

# # Загрузка данных
x = "/home/genetics/ML/" # Папка назначения для сохранения выходных данных
data = pd.read_excel("/home/genetics/ML/ML_model 19.02.24.xlsx") # Местоположение датасета

# Модификация данных

data = data.rename(columns = {'Гистотип: аденокарцинома -2, нейроэндоксринная -1, доброкач - 0' : 'Морфологический тип опухоли', 'Амилаза крови 1-15 п/о дни': 'Амилаза крови 10-15 п/о дни'})

# Удаление и восполнение пропущенных данных 
data['Возраст'].fillna(data['Возраст'].mean(), inplace=True)
data['Время операции, мин'].fillna(data['Время операции, мин'].mean(), inplace=True)
data['Объем кровопотери, мл'].fillna(data['Объем кровопотери, мл'].mean(), inplace=True)
data['Продолжительность дренирования, сут'].fillna(data['Продолжительность дренирования, сут'].mean(), inplace=True)
data['ИМТ'].fillna(data['ИМТ'].mean(), inplace=True)
data['Размеры опухоли, см3'].fillna(data['Размеры опухоли, см3'].mean(), inplace=True)
data['Амилаза дренажа 1-5 п/о дни'].fillna(data['Амилаза дренажа 1-5 п/о дни'].mean(), inplace=True)
data['Амилаза дренажа 10-15 п/о дни'].fillna(data['Амилаза дренажа 10-15 п/о дни'].mean(), inplace=True)
data['Лейкоциты, ОАК 5-7 п/о дни'].fillna(data['Лейкоциты, ОАК 5-7 п/о дни'].mean(), inplace=True)
data['Лейкоциты, ОАК 10-15 п/о дни'].fillna(data['Лейкоциты, ОАК 10-15 п/о дни'].mean(), inplace=True)
data['Пол'].fillna(data['Пол'].mode()[0], inplace=True)
data['Резекция сосудов'].fillna(data['Резекция сосудов'].mode()[0], inplace=True)
data['Размер  Вирсунгова протока КТ, мм'].fillna(data['Размер  Вирсунгова протока КТ, мм'].mode()[0], inplace=True)
data['Анастомоз по Блюмгарту '].fillna(data['Анастомоз по Блюмгарту '].mode()[0], inplace=True)
data['Анастомоз по РУ'].fillna(data['Анастомоз по РУ'].mode()[0], inplace=True)
data['Пункционная энтеростома'].fillna(data['Пункционная энтеростома'].mode()[0], inplace=True)
data['НАПХТ'].fillna(data['НАПХТ'].mode()[0], inplace=True)
data['Морфологический тип опухоли'].fillna(data['Морфологический тип опухоли'].mode()[0], inplace=True)
data['G'].fillna(data['G'].mode()[0], inplace=True)
data['R-cтатус '].fillna(data['R-cтатус '].mode()[0], inplace=True)
data['Стадия'].fillna(data['Стадия'].mode()[0], inplace=True)
data['Периневральная инвазия'].fillna(data['Периневральная инвазия'].mode()[0], inplace=True)
data['Васкулярная инвазия'].fillna(data['Васкулярная инвазия'].mode()[0], inplace=True)
data['Послеоперационная смерть'].fillna(data['Послеоперационная смерть'].mode()[0], inplace=True)
data.loc[(data['Лейкоциты, ОАК 1-5 п/о дни'] == ' 8,9 _x000D_\n'), ['Лейкоциты, ОАК 1-5 п/о дни']] = None
data['Лейкоциты, ОАК 1-5 п/о дни'].astype('float')
data['Лейкоциты, ОАК 1-5 п/о дни'].fillna(data['Лейкоциты, ОАК 1-5 п/о дни'].mean(), inplace=True)

data.loc[(data['Лейкоциты, ОАК 1-3 п/о дни'] == ' 8,9 _x000D_\n'), ['Лейкоциты, ОАК 1-3 п/о дни']] = None
data['Лейкоциты, ОАК 1-3 п/о дни'].astype('float')
data['Лейкоциты, ОАК 1-3 п/о дни'].fillna(data['Лейкоциты, ОАК 1-3 п/о дни'].mean(), inplace=True)

data['Амилаза крови 1-5 п/о дни'].fillna(data.query('`Амилаза крови 1-5 п/о дни`  < 120')['Амилаза крови 1-5 п/о дни'].mean(), inplace=True)
data['Амилаза крови 10-15 п/о дни'].fillna(data.query('`Амилаза крови 10-15 п/о дни`  < 120')['Амилаза крови 10-15 п/о дни'].mean(), inplace=True)

# Удаления параметров для в рамках реализации возможности их выбора для моделирования
data = data.drop(['R-cтатус ' , 
           'Продолжительность дренирования, сут', 'Послеоперационная смерть', 'Лейкоциты, ОАК 1-3 п/о дни'], axis = 1)


# Корреляция кендала
fig, ax = plt.subplots(figsize=(9, 9))
seaborn.heatmap(data.corr(method='kendall'), annot = True, fmt='.1g', cmap= 'coolwarm')
None

# Подготовка входных данных 
num_columns = ['Возраст',
              'Время операции, мин',
              'Объем кровопотери, мл',
              #'Продолжительность дренирования, сут',
              'ИМТ',
              'Размеры опухоли, см3',
              'Амилаза крови 1-5 п/о дни',
              'Амилаза крови 10-15 п/о дни',
              'Амилаза дренажа 1-5 п/о дни',
              'Амилаза дренажа 10-15 п/о дни' ,
              # 'Лейкоциты, ОАК 1-3 п/о дни' ,
              'Лейкоциты, ОАК 1-5 п/о дни' ,
              'Лейкоциты, ОАК 5-7 п/о дни',
              'Лейкоциты, ОАК 10-15 п/о дни'
              ]
cat_columns = ['Пол',
              'Резекция сосудов',
              'Размер  Вирсунгова протока КТ, мм',
              'Анастомоз по Блюмгарту ',
              'Анастомоз по РУ',
              'Пункционная энтеростома',
               'НАПХТ',
               'Стадия',
               'Морфологический тип опухоли',
               'G',
               #'R-cтатус ',
               'Периневральная инвазия',
               'Васкулярная инвазия',
               #'Послеоперационная смерть'
              ]

# # Пайплайн

X = data.drop(['Послеоперационные фистулы 2 класса', 'Послеоперационные фистулы В/С', 'Послеоперационные фистулы бх+В/С'], axis=1)
y = data['Послеоперационные фистулы бх+В/С']

columns_names = cat_columns + num_columns

X_train, X_test, y_train, y_test = train_test_split(
    data.drop(['Послеоперационные фистулы 2 класса', 'Послеоперационные фистулы В/С', 'Послеоперационные фистулы бх+В/С'], axis=1),
    data['Послеоперационные фистулы бх+В/С'],
    test_size = 0.25, 
    random_state = 42,
    stratify = data['Послеоперационные фистулы бх+В/С'])

encoder = LabelEncoder()
y_train_trans = encoder.fit_transform(y_train)
y_test_trans = encoder.transform(y_test)

data_preprocessor = ColumnTransformer(
    [
        ('cat', StandardScaler(), cat_columns),
        ('num', StandardScaler(), num_columns)
    ], 
    remainder='passthrough'
) 

pipe_final = Pipeline(
    [
        ('preprocessor', data_preprocessor),
        ('models', DecisionTreeClassifier(random_state=42))
    ]
) 


param_grid = [
    # словарь для модели DecisionTreeClassifier()
  {
    'models': [DecisionTreeClassifier(random_state=42)],
    'models__max_depth': [None, 1, 3, 5, 10, 20, 30],
    'models__max_features': range(1, 28)
},
    
   # словарь для модели KNeighborsClassifier() 
    {
       'models': [KNeighborsClassifier()],
       'models__n_neighbors': range(2, 5) 
   },
    
       # словарь для модели RandomForestClassifier() 
    {
       'models': [RandomForestClassifier(random_state=42)],
        'models__n_estimators': [50, 100, 200, 150, 500, 1000],
        'models__max_depth': [None, 1, 3, 5, 10, 20, 30],
        'models__min_samples_leaf': [1, 2, 4, 6, 8],
        'models__max_features': range(1, 28)
   },
    
           # словарь для модели CatBoostClassifier() 
    {
       'models': [CatBoostClassifier()],
        'models__depth': [4,5,6,7,8,9, 10],
        'models__learning_rate' : [0.01,0.02,0.03,0.04, 0.001, 0.0001, 0.1],
        'models__iterations'    : [10, 20,30,40,50,60,70,80,90, 100, 500]
   },
    
    # словарь для модели SVC() – метод опорных векторов
    
        {
        'models': [SVC(probability=True, random_state=42)],
        'models__C':[0.1,1,10,50,100],
        'models__kernel':['rbf','poly','sigmoid','linear'],
        'models__degree':[1,2,3,4,5,6],
         'models__gamma': [1, 0.1, 0.01, 0.001, 0.0001]
    },

    # словарь для модели LogisticRegression() – логистическая регрессия
    {
        'models': [LogisticRegression(
            random_state=42, 
            solver='liblinear', 
        )],
        'models__C': range(1, 10),
        'models__penalty': ['l1','l2'],
        'models__max_features': range(1, 28)
    },
    
        # словарь для модели Ridge() – гребневая регрессия
    {
        'models': [Ridge()],
        "models__alpha": [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20],
        'models__max_features': range(1, 28)
    }
]

randomized_search = GridSearchCV(
    pipe_final, 
    param_grid, 
    cv=5,
    scoring='roc_auc',
    #random_state=43,
    n_jobs=-1
)
randomized_search.fit(X, y)

print('Лучшая модель и её параметры:\n\n', randomized_search.best_estimator_)
print ('Метрика лучшей модели на тренировочной выборке:', randomized_search.best_score_)
print('Метрика лучшей модели на тестовой выборке:', roc_auc_score(randomized_search.best_estimator_.predict(X_test), y_test_trans))

# ### Лучшая модель
X_train_get = pd.DataFrame(data_preprocessor.fit_transform(X_train))
X_test_get = pd.DataFrame(data_preprocessor.transform(X_test))
best_model = RandomForestClassifier(max_depth=3, max_features=9,
                                        min_samples_leaf=4, n_estimators=50,
                                        random_state=42)
best_model.fit(X_train_get, y_train_trans)
X_train_get.columns = [columns_names]
X_test_get.columns = [columns_names]

