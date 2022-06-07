from turtle import color
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris


# load the data with load_iris from sklearn
data = load_iris()

features = data['data']
feature_names = data['feature_names']
target = data['target']
target_names = data['target_names']


## Plotting the sepal width and length
fig, ax = plt.subplots()
for t,marker,c in zip([0,1,2],">ox","rgb"):
    # We plot each class on its own to get different colored markers
    scatter = ax.scatter(features[target == t, 0], 
    features[target == t, 1],
    label=target_names[t],
    marker=marker,
    color=c)

ax.legend()
plt.grid(True)
plt.show()

## Ploting the pental length and height
fig, ax = plt.subplots()
for t,marker,c in zip([0,1,2],">ox","rgb"):
    # We plot each class on its own to get different colored markers
    scatter = ax.scatter(features[target == t, 2], 
    features[target == t, 3],
    label=target_names[t],
    marker=marker,
    color=c)

ax.legend()
plt.grid(True)
plt.show()


plength = features[:, 2]
# use numpy operations to get setosa features
is_setosa = (target == 0)
# This is the important step:
max_setosa =plength[is_setosa].max()
min_non_setosa = plength[~is_setosa].min()
print('Maximum of setosa: {0}.'.format(max_setosa))
print('Minimum of others: {0}.'.format(min_non_setosa))



