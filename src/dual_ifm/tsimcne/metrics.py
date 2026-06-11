from annoy import AnnoyIndex
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


def ann_acc(
    X_train,
    X_test,
    y_train,
    y_test,
    n_neighbors=15,
    metric='cosine',
    n_trees=20,
    seed=10106122,
):
    metric = metric if metric != 'cosine' else 'angular'
    nn = AnnoyIndex(X_test.shape[1], metric)
    nn.set_seed(seed)

    for i, x in enumerate(X_train):
        nn.add_item(i, x)
    nn.build(n_trees)

    nn_ixs = [nn.get_nns_by_vector(x, n_neighbors) for x in X_test]
    preds, _ = stats.mode(y_train[nn_ixs], axis=1, keepdims=False)

    return (preds == y_test).mean()


def knn_acc(
    X_train,
    X_test,
    y_train,
    y_test,
    n_neighbors=15,
    metric='euclidean',
    n_jobs=-1,
    **kwargs,
):
    knn = KNeighborsClassifier(n_neighbors, metric=metric, n_jobs=n_jobs, **kwargs)

    knn.fit(X_train, y_train)
    return knn.score(X_test, y_test)


def knn_reg(
    X_train,
    X_test,
    y_train,
    y_test,
    n_neighbors=15,
    metric='euclidean',
    n_jobs=-1,
    **kwargs,
):
    knn = KNeighborsRegressor(n_neighbors, metric=metric, n_jobs=n_jobs, **kwargs)

    knn.fit(X_train, y_train)
    return knn.score(X_test, y_test)


def linear_acc(X_train, X_test, y_train, y_test, solver='saga', n_jobs=-1, **kwargs):
    lin = LogisticRegression(solver=solver, n_jobs=n_jobs, **kwargs)

    lin.fit(X_train, y_train)
    return lin.score(X_test, y_test)


def linear_reg(X_train, X_test, y_train, y_test, n_jobs=-1, **kwargs):
    lin = LinearRegression(n_jobs=n_jobs, **kwargs)

    lin.fit(X_train, y_train)
    return lin.score(X_test, y_test)
