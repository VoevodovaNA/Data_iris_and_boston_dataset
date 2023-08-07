import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

from scipy.stats import kstest

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from scipy.stats import sem
from scipy import polyval, stats

from sklearn.datasets import load_boston
boston_dataset = load_boston()
pd.set_option("display.max.columns", None)
boston = pd.DataFrame(data=boston_dataset['data'], columns=boston_dataset['feature_names'])
boston['MEDV'] = boston_dataset['target']
y = boston['MEDV'].copy()
#y =  np.log1p(y)
del boston['MEDV']
boston = pd.concat((y, boston), axis=1)

print(boston.head())

boston = (boston - np.min(boston)) / (np.max(boston) - np.min(boston))
print(boston)


import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
#fig = plt.figure()
#ax1 = fig.add_subplot(1,1,1)
#plt.boxplot(x=boston)
#plt.title('Диаграмма Тьюки')
#plt.show()

#print(boston.isnull().sum())

#fig = plt.figure(figsize=(60, 60))
#fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
#index = 0
#axs = axs.flatten()
#for k,v in boston.items():
#    sns.distplot(v, ax=axs[index])
#    index += 1
#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
#sns.set(rc={'figure.figsize':(11.7,8.27)})
#sns.distplot(boston['MEDV'], bins=30)
#plt.show()

#correlation_matrix = boston.corr().round(2)
# annot = True to print the values inside the square
#sns.heatmap(data=correlation_matrix, annot=True)
#plt.show()
#fig = plt.figure()
#ax1 = fig.add_subplot(1,1,1)
#plt.figure(figsize=(10, 10))
#sns.heatmap(boston.corr().abs(),  annot=True)
#plt.show()

from scipy.stats import kstest
def test_kolm(arr):
    print('MEDV:   ')
    print(kstest(arr['MEDV'], 'norm'))
    print('CRIM:   ')
    print(kstest(arr['CRIM'], 'norm'))
    print('ZN:   ')
    print(kstest(arr['ZN'], 'norm'))
    print('INDUS:   ')
    print(kstest(arr['INDUS'], 'norm'))
    print('CHAS:   ')
    print(kstest(arr['CHAS'], 'norm'))
    print('NOX:   ')
    print(kstest(arr['NOX'], 'norm'))
    print('RM:   ')
    print(kstest(arr['RM'], 'norm'))
    print('AGE:   ')
    print(kstest(arr['AGE'], 'norm'))
    print('DIS:   ')
    print(kstest(arr['DIS'], 'norm'))
    print('RAD:   ')
    print(kstest(arr['RAD'], 'norm'))
    print('TAX:   ')
    print(kstest(arr['TAX'], 'norm'))
    print('PTRATIO:   ')
    print(kstest(arr['PTRATIO'], 'norm'))
    print('B:   ')
    print(kstest(arr['B'], 'norm'))
    print('LSTAT:   ')
    print(kstest(arr['LSTAT'], 'norm'))

def norm_rasp_ost(arr):
    print(kstest(arr, 'norm'))

#'MEDV','LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE'
from sklearn import preprocessing
# Let's scale the columns before plotting them against MEDV
#min_max_scaler = preprocessing.MinMaxScaler()
#column_sels = ['MEDV','CRIM','ZN','LSTAT', 'CHAS','RAD','B','INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']
#x = boston.loc[:,column_sels]
#y = boston['MEDV']
#x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=column_sels)
#print(boston)
#fig = plt.figure(figsize=(60, 60))
#fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))
#index = 0
#axs = axs.flatten()
#for i, k in enumerate(column_sels):
#    sns.regplot(y=y, x=x[k], ax=axs[i])
#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
#for i, k in enumerate(column_sels):
#    sns.residplot(y=y, x=x[k], ax=axs[i])
#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
#plt.show()

#y =  np.log1p(y)
#print(x)
#print(y)

#fig = plt.figure()
#ax1 = fig.add_subplot(1,1,1)
#plt.boxplot(x=boston)
#plt.title('Диаграмма Тьюки')
#plt.show()
def rescale(X):
    mean = X.mean()
    std = X.std()
    scaled_X = [(i - mean)/std for i in X]
    return pd.Series(scaled_X)
#df_std = pd.DataFrame(columns=df.columns)
#for i in df.columns:
#    df_std[i] = rescale(df[i])

#frames = [x, y]
#x = pd.concat(frames)
from sklearn.metrics import mean_squared_error
def mn_reg_plot(arr):
    X_train, X_test, y_train, y_test = train_test_split(
        arr[['CRIM','ZN','LSTAT', 'CHAS','RAD','B','INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']], arr['MEDV'],
        test_size=0.3, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    Y_pred = model.predict(X_test)
    #y = arr['MEDV']
    #resid_MEDV = y-Y_pred
    a = model.intercept_
    b = model.coef_
    #print("Лучшая линия: отрезок", a, ", коэффициент регрессии:", b)
    score1 = model.score(X_test, y_test)
    score2 = model.score(X_train, y_train)
    print('коэффицент детерминации test=', score1)
    print('коэффицент детерминации train=', score2)
    print(Y_pred)
    print(y_test)
    residuals = y_test - Y_pred
    # histogram plot
    residuals.hist()
    plt.show()
    # density plot
    residuals.plot(kind='kde')
    plt.show()

    #ostatki = y_test - Y_pred
    #error = mean_squared_error(y, Y_pred)
    #print('std error', sem(Y_pred))
    # СКО - средний квадрат отклонения
    #print('mse = {:.2f}'.format(error))
    # Гипотеза о нормальном распределении остатоков - нет норм распред
    #norm_rasp_ost(ostatki)
    #print(len(Y_pred))
    #print(X_test)
    #plt.subplot(1, 1, 1)
    #plt.plot(range(len(Y_pred)), Y_pred, 'b', label='predict')
    #plt.subplot(1, 1, 1)
    #plt.plot(range(len(y_test)), y_test, 'r', label="test")
    #plt.subplot(1, 1, 1)
    #plt.plot(range(len(Y_pred - y_test)), Y_pred - y_test, 'g', label="ostatki")
    #plt.legend()
    #plt.show()
    #fig, ax = plt.subplots(figsize=(5, 7))
    #ax.scatter(x, resid_MEDV, alpha=0.6)
    #ax.set_xlabel('LSTAT')
    #ax.set_ylabel('MEDV Residual $(y-\hat{y})$')
    #plt.axhline(0, color='black', ls='dotted')
    #plt.show()
mn_reg_plot(boston)

from sklearn.manifold import TSNE

#tsne = TSNE(n_components=2)
#tsne_Components = tsne.fit_transform(boston)
#tsne_Df = pd.DataFrame(data = tsne_Components
#             , columns = ['Component 1', 'Component 2'])
#finalDf = pd.concat([tsne_Df, boston[['CHAS']]], axis = 1)
#fig = plt.figure()
#ax = fig.add_subplot(1,1,1)
#ax.set_xlabel('Principal Component 1', fontsize = 15)
#ax.set_ylabel('Principal Component 2', fontsize = 15)
#ax.scatter(tsne_Df['Component 1'], tsne_Df['Component 2'])

#plt.title("t-sne")
#plt.show()
#print(finalDf)

from sklearn.decomposition import PCA
#pca = PCA(n_components=13)
#X = boston.drop('MEDV',axis=1)
#X_pca = pca.fit_transform(X)
#df_std_pca = pd.DataFrame(X_pca,columns=['PCA1','PCA2','PCA3','PCA4','PCA5','PCA6','PCA7','PCA8','PCA9','PCA10','PCA11','PCA12','PCA13'])
#df_std_pca['MEDV'] = boston['MEDV']
#fig = plt.figure(figsize=(15,8))
#ax = fig.add_subplot(111)
#sns.heatmap(df_std_pca.corr(),annot=True)
#plt.show()

from sklearn.decomposition import PCA
#pca = PCA(n_components=2)
#principalComponents = pca.fit_transform(data)
#principalDf = pd.DataFrame(data = principalComponents
#             , columns = ['principal component 1', 'principal component 2'])
#fig = plt.figure(figsize = (8,8))
#ax = fig.add_subplot(1,1,1)
#ax.set_xlabel('Principal Component 1', fontsize = 15)
#ax.set_ylabel('Principal Component 2', fontsize = 15)
#ax.set_title('2 component PCA', fontsize = 20)
#colors = ['r', 'g', 'b']
#ax.scatter(principalDf['principal component 1'], principalDf['principal component 2'])
#ax.grid()

from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist

#distortions = []
#inertias = []
#mapping1 = {}
#mapping2 = {}
#K = range(1, 10)

#for k in K:
    # Building and fitting the model
#    kmeanModel = KMeans(n_clusters=k).fit(boston)
#    kmeanModel.fit(boston)

#    distortions.append(sum(np.min(cdist(boston, kmeanModel.cluster_centers_,
#                                        'euclidean'), axis=1)) / boston.shape[0])
 #   inertias.append(kmeanModel.inertia_)

#    mapping1[k] = sum(np.min(cdist(boston, kmeanModel.cluster_centers_,
#                                   'euclidean'), axis=1)) / boston.shape[0]
#    mapping2[k] = kmeanModel.inertia_
#for key, val in mapping1.items():
#    print(f'{key} : {val}')
#plt.plot(K, distortions, 'bx-')
#plt.xlabel('Values of K')
#plt.ylabel('Distortion')
#plt.title('Метод локтя')
#plt.show()


#tsne = TSNE(n_components=2)
#tsne_Components = tsne.fit_transform(boston)
#tsne_Df = pd.DataFrame(data = tsne_Components
#             , columns = ['Component 1', 'Component 2'])
#finalDf = pd.concat([tsne_Df, boston[['CHAS']]], axis = 1)
#fig = plt.figure()
#ax = fig.add_subplot(1,1,1)
#ax.set_xlabel('Principal Component 1', fontsize = 15)
#ax.set_ylabel('Principal Component 2', fontsize = 15)
#ax.scatter(tsne_Df['Component 1'], tsne_Df['Component 2'])

#plt.title("t-sne")
#plt.show()
#print(finalDf)

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans
from numpy import sqrt, array, random, argsort
from sklearn.preprocessing import scale
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
# n_jobs is the number of CPU core to run parallel

def tsnee(boston):

    # Определяем модель и скорость обучения
    model = KMeans(n_clusters=7)
    # Обучаем модель
    transformed = model.fit_transform(boston_dataset['data'])

    # Представляем результат в двумерных координатах
    x_axis = transformed[:, 0]
    y_axis = transformed[:, 1]

    plt.scatter(x_axis, y_axis, c=boston_dataset['target'])
    plt.title('Проекция с помощью tsne')
    plt.show()

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(boston)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax.set_xlabel('Pr Comp 1', fontsize = 15)
#ax.set_ylabel('Pr Comp 2', fontsize = 15)
#ax.set_zlabel('Pr Comp 3', fontsize = 15)
#colors = ['r', 'g', 'b']
#ax.scatter(principalDf['principal component 1'], principalDf['principal component 2'], principalDf['principal component 3'])
#ax.grid()
#plt.title("pca")
#plt.show()
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=7)
kmeans.fit(boston)
y_kmeans = kmeans.predict(boston)
plt.scatter(principalDf['principal component 1'], principalDf['principal component 2'], c=y_kmeans, s=5, cmap='viridis')
plt.title("К-средних")
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], s=5, alpha=0.5)
plt.show()

