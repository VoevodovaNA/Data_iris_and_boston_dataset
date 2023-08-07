import pingouin as pg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from scipy.stats import kstest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

boston_dataset = load_boston()

boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target
print(boston)

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.displot(boston['MEDV'], bins=30)
plt.show()

plt.boxplot(boston)
plt.title('Диаграмма Тьюки')
plt.show()

sns.heatmap(boston.corr(), annot=True)
plt.show()

boston.isnull().sum
print(boston.describe())

sns.pairplot(boston)
plt.show()
fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in boston.items():
    sns.distplot(v, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.show()

from sklearn import preprocessing
# Let's scale the columns before plotting them against MEDV
min_max_scaler = preprocessing.MinMaxScaler()
column_sels = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']
x = boston.loc[:,column_sels]
y = boston['MEDV']
x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=column_sels)
fig, axs = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for i, k in enumerate(column_sels):
    sns.regplot(y=y, x=x[k], ax=axs[i])
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
plt.show()

scaler = StandardScaler()
scaler.fit(boston)
StandardScaler(copy=True, with_mean=True, with_std=True)
scaled_data = scaler.transform(boston)
print('scaled_data=', scaled_data)

X = boston[['RM', 'LSTAT', 'PTRATIO']].values
y = boston['MEDV'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
lm = LinearRegression()
lm.fit(X_train,y_train)
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show()
print('коэффицент детерминации', lm.score(X_train, y_train))
print('коэффицент детерминации', lm.score(X_test, y_test))
score = lm.score(X, y)

X=scaled_data[:,0:13]
y=scaled_data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
mean=X_train.mean(axis=0)
std.X_train.std(axis=0)
X_train -= mean
X_train /= std
X_test -= mean
X_test /= std
print(X_test)
model = LinearRegression()
model.fit(X, y)
Y_pred = model.predict(X)
a = model.intercept_
b = model.coef_
print("Лучшая линия: отрезок", a, ", коэффициент регрессии:", b)
remains = y - Y_pred
# Гипотеза о нормальном распределении остатоков - нет норм распред
# norm_rasp_ost(ostatki)
plt.subplot(2, 3, 1)
plt.scatter(boston['RM'], Y_pred, label='predict')
plt.scatter(boston['RM'], y, label='data')
plt.title('Предсказание по RM')
plt.subplot(2, 3, 2)
plt.scatter(boston['LSTAT'], Y_pred)
plt.scatter(boston['LSTAT'], y)
plt.title('Предсказание по LSTAT')
plt.subplot(2, 3, 3)
plt.scatter(boston['PTRATIO'], Y_pred)
plt.scatter(boston['PTRATIO'], y)
plt.title('Предсказание по PTRATIO')
plt.subplot(2, 3, 4)
plt.scatter(y, remains, label='remains')
plt.legend()
plt.show()
score = model.score(X_train, y_train)
print('коэффицент детерминации=', score)

print(kstest(remains, 'norm'))
score = model.score(X_train, y_train)
plt.scatter(y_train,Y_pred)
plt.xlabel('Prices')
plt.ylabel('Predict prices')
plt.title('Prices vs Predict prices')
plt.show()
print('коэффицент детерминации=', score)

scaler = StandardScaler()
scaler.fit(boston)
StandardScaler(copy=True, with_mean=True, with_std=True)
scaled_data = scaler.transform(boston)
pca = PCA(n_components=2)
pca.fit(scaled_data)
PCA(copy=True, iterated_power='auto', n_components=3, random_state=None, svd_solver='auto', tol=0.0, whiten=False)
x_pca = pca.transform(scaled_data)
print(scaled_data.shape)
print(x_pca.shape)
plt.scatter(x_pca[:,0],x_pca[:,1])
plt.title('PCA')
plt.show()

from sklearn.manifold import TSNE
model = TSNE(n_components=2, perplexity=10, learning_rate='auto', n_iter=3000)
tsne_data = model.fit_transform(scaled_data)
print(tsne_data.shape)
plt.scatter(tsne_data[:,0], tsne_data[:,1])
plt.title('t-SNE')
plt.show()

model = TSNE(n_components=2, perplexity=10, learning_rate='auto', n_iter=3000)
tsne_data = model.fit_transform(boston)
print(tsne_data.shape)
plt.scatter(tsne_data[:,0], tsne_data[:,1])
plt.title('t-SNE')
plt.show()

pca = PCA(n_components=2)
pca.fit(boston)
PCA(copy=True, iterated_power='auto', n_components=3, random_state=None, svd_solver='auto', tol=0.0, whiten=False)
x_pca = pca.transform(boston)
print(boston.shape)
print(x_pca.shape)
plt.scatter(x_pca[:,0],x_pca[:,1])
plt.title('PCA')
plt.show()

def K_Means_iris(array):
    x_=array
    no_of_clusters = [2,3,4]
    for n_clusters in no_of_clusters:
        model = KMeans(n_clusters=n_clusters) #, init='k-means++', random_state=2020)
        model.fit(x_)
        #all_predictions = model.predict(x_)
        fig = plt.figure()
        ax_3d = fig.add_subplot(projection='3d')
        ax_3d.scatter(['sepal width (cm)'], iris_pd['petal length (cm)'], iris_pd['petal width (cm)'], c=kmeans.labels_, cmap='rainbow')
        ax_3d.set_xlabel('sepal length')
        ax_3d.set_ylabel('petal length')
        ax_3d.set_zlabel('petal width')
        plt.title('Центроидный метод k-средних(7 кластеров)')
        plt.show()
        #print('n_clusters = ',n_clusters)
        #print(all_predictions)

from scipy.cluster.hierarchy import dendrogram, linkage
x = boston.iloc[:, 0].values
y = boston.iloc[:, 1].values
z = boston.iloc[:, 2].values
data = list(zip(x,y,z))
linkage_data = linkage(data,method='single',metric='euclidean')
dendrogram(linkage_data)
plt.title('Метод: Одиночная связь, Расстояние: Евклидово расстояние')
plt.show()
linkage_data = linkage(data,method='complete',metric='euclidean')
dendrogram(linkage_data)
plt.title('Метод: Полная связь, Расстояние: Евклидово расстояние')
plt.show()
linkage_data = linkage(data,method='complete',metric='chebychev')
dendrogram(linkage_data)
plt.title('Метод: Полная связь, Расстояние: Расстояние Чебышева')
plt.show()
linkage_data = linkage(data,method='complete',metric='cityblock')
dendrogram(linkage_data)
plt.title('Метод: Полная связь, Расстояние: Манхэтэнское расстояние')
plt.show()