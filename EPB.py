import random

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.preprocessing import normalize
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, LinearRegression, ARDRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings

warnings.filterwarnings("ignore")
train_pr = 0.75

raw_data = pd.read_csv(open('ENB2012_data.csv'))
raw_data = raw_data.values[0:768, 0:10]
index = [i for i in range(len(raw_data))]
random.shuffle(index)
raw_data = raw_data[index]
raw_data = normalize(raw_data, axis=0, norm='max')
x = raw_data[:, 0:8]

y1 = raw_data[:, 8:9].reshape(-1, 1)
y2 = raw_data[:, 9:10].reshape(-1, 1)
train_index = int(train_pr * len(x))
x_train = x[0:train_index]
y1_train = y1[0:train_index]
y2_train = y2[0:train_index]

x_test = x[train_index:]
y1_test = y1[train_index:]
y2_test = y2[train_index:]


def mape(clf, x, y):
    clf.fit(x, y)
    y_pred = clf.predict(x).flatten()
    sigma = 0
    for i in range(len(y_pred)):
        sigma += (np.abs(y.flatten()[i] - y_pred[i]) / [y.flatten()[i]])
    sigma *= (100 / len(x))
    return sigma


def mae(clf, x, y):
    clf.fit(x, y)
    y_pred = clf.predict(x)
    error = np.absolute(y_pred.flatten() - y.flatten())
    return np.mean(error)


def rmse(clf, x, y):
    clf.fit(x, y)
    y_pred = clf.predict(x)
    error = y_pred.flatten() - y.flatten()
    for i in range(len(error)):
        error[i] = error[i] ** 2
    return np.sqrt(np.mean(error))


# clf = SVR(kernel='rbf', gamma='auto', C=200, epsilon=0)

clf = GradientBoostingRegressor()
# clf = RandomForestRegressor(max_features='auto', n_estimators=200)
# clf.fit(x_train, y1_train)
# print(clf.score(x_test, y1_test))
#############################################
print('Heating results...')
print('RMSE')
crs = cross_val_score(clf, x, y1, cv=10, scoring=rmse)
print('avg', crs.mean(), 'std', crs.std())
## convert your array into a dataframe
df = pd.DataFrame(crs)

## save to xlsx file

filepath = 'heating_RMSE.xlsx'

df.to_excel(filepath, index=False)
print('MAE')
crs = cross_val_score(clf, x, y1, cv=10, scoring=mae)
print('avg', crs.mean(), 'std', crs.std())
## convert your array into a dataframe
df = pd.DataFrame(crs)

## save to xlsx file

filepath = 'heating_MAE.xlsx'

df.to_excel(filepath, index=False)
print('R2')
crs = cross_val_score(clf, x, y1, cv=10)
print('avg', crs.mean(), 'std', crs.std())
df = pd.DataFrame(crs)

## save to xlsx file

filepath = 'heating_R2.xlsx'

df.to_excel(filepath, index=False)
print('MAPE')
crs = cross_val_score(clf, x, y1, cv=10, scoring=mape)
df = pd.DataFrame(crs)

## save to xlsx file

filepath = 'heating_MAPE.xlsx'

df.to_excel(filepath, index=False)
print('avg', crs.mean(), 'std', crs.std())
####################################################
print('Cooling results...')
print('RMSE')
crs = cross_val_score(clf, x, y2, cv=10, scoring=rmse)
df = pd.DataFrame(crs)

## save to xlsx file

filepath = 'Cooling_RMSE.xlsx'

df.to_excel(filepath, index=False)
print('avg', crs.mean(), 'std', crs.std())
print('MAE')
crs = cross_val_score(clf, x, y2, cv=10, scoring=mae)
df = pd.DataFrame(crs)

## save to xlsx file

filepath = 'Cooling_MAE.xlsx'

df.to_excel(filepath, index=False)
print('avg', crs.mean(), 'std', crs.std())
print('R2')
crs = cross_val_score(clf, x, y2, cv=10)
df = pd.DataFrame(crs)

## save to xlsx file

filepath = 'Cooling_R2.xlsx'

df.to_excel(filepath, index=False)
print('avg', crs.mean(), 'std', crs.std())
print('MAPE')
crs = cross_val_score(clf, x, y2, cv=10, scoring=mape)
df = pd.DataFrame(crs)

## save to xlsx file

filepath = 'Cooling_MAPE.xlsx'

df.to_excel(filepath, index=False)
print('avg', crs.mean(), 'std', crs.std())
############################################## clustering
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import MiniBatchKMeans, KMeans

n_clusters_list = [3, 4, 5, 6, 7, 8]
X = np.array([y1.flatten(), y2.flatten()]).T
# Algorithms to compare
clustering_algorithms = {
    "MiniBatchKMeans": MiniBatchKMeans,
    "K-Means": KMeans,
}

# Make subplots for each variant
fig, axs = plt.subplots(
    len(clustering_algorithms), len(n_clusters_list), figsize=(15, 5)
)

axs = axs.T
distances = {}

for i, (algorithm_name, Algorithm) in enumerate(clustering_algorithms.items()):
    distances[algorithm_name] = {'Inertia': [], 'Silhouette': []}
    for j, n_clusters in enumerate(n_clusters_list):
        algo = Algorithm(n_clusters=n_clusters)
        algo.fit(X)
        centers = algo.cluster_centers_
        print(algorithm_name, 'number of clusters:', n_clusters, 'objective function:', algo.inertia_)
        axs[j, i].scatter(X[:, 0], X[:, 1], s=10, c=algo.labels_)
        distances[algorithm_name]['Inertia'].append(algo.inertia_)
        distances[algorithm_name]['Silhouette'].append(metrics.silhouette_score(X, algo.predict(X)))
        axs[j, i].scatter(centers[:, 0], centers[:, 1], c="r", s=20)

        axs[j, i].set_title(f"{algorithm_name} : {n_clusters} clusters")

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
plt.clf()
plt.cla()

plt.plot(n_clusters_list, distances["MiniBatchKMeans"]['Inertia'], 'bx-')
plt.plot(n_clusters_list, distances["K-Means"]['Inertia'], 'rx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.legend(['MiniBatchKMeans', 'K-Means'])
plt.show()

plt.clf()
plt.clf()
plt.plot(n_clusters_list, distances["MiniBatchKMeans"]['Silhouette'], 'bx-')
plt.plot(n_clusters_list, distances["K-Means"]['Silhouette'], 'rx-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method For Optimal k')
plt.legend(['MiniBatchKMeans', 'K-Means'])
plt.show()
