import numpy as np
from sklearn.linear_model import LinearRegression
from build_1N2_from_history import  load_compute_matrices, prepare_logistic_regression
from sklearn.metrics import mean_squared_error, r2_score
import itertools

n_cat = 20
stats, rebuilt, filtered = load_compute_matrices(
    1900, 2020,
    threshold_1=100,
    threshold_2=150,
    filter_threshold=100,
    n_cat=n_cat,
    from_file='paris_sportifs_filtered.csv'
)

data = np.zeros((len(stats), 3))

index = 0
for (ht, at), r in stats.items():
    v = r['p'][2]
    data[index, 0] = ht
    data[index, 1] = at
    data[index, 2] = v
    index += 1

data_X = data[:, :2]
data_Y = data[:, 2]

regr = LinearRegression()
regr.fit(data_X, data_Y)

print(regr.coef_)
print(regr.intercept_)
data_Y_pred = regr.predict(data_X)

print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(data_Y, data_Y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(data_Y, data_Y_pred))


n_cat = 20
stats = prepare_logistic_regression(
    1900, 2020,
    n_cat=n_cat,
    from_file='paris_sportifs_filtered.csv'
)

data = np.array(stats)
X = data[:, :2]
y = data[:, 2]

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, y)

ra = range(n_cat)
print("Aa,Ba,gd,p")
for Aa, Ba in itertools.product(ra, ra):
    p = clf.predict_proba([[Aa, Ba]])
    [print("{},{},{},{}".format(Aa, Ba, i, p[0][i])) for i in range(3)]
