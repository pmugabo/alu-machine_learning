# Mean, Variance, Covariance, Correlation

## Mean
```python
def mean(x):
    # x is a vector
    return sum(x) / len(x)
```
    
## Variance
```python
def variance(x):
    mu = mean(x)
    return sum([(xi - mu) ** 2 for xi in x]) / len(x)
```

## Covariance
```python
def covariance(x, y):
    n = len(x)
    return dot(de_mean(x), de_mean(y)) / (n - 1)
```

## Correlation
```python
def correlation(x, y):
    stdev_x = sqrt(variance(x))
    stdev_y = sqrt(variance(y))
    if stdev_x > 0 and stdev_y > 0:
        return covariance(x, y) / stdev_x / stdev_y
    else:
        return 0
def correlatio(x)
    # x is a covariance matrix
    # loop through the rows and columns
    # and calculate the correlation
    # return a correlation matrix
    for i in range(len(x)):
        for j in range(len(x)):
            x[i][j] = x[i][j] / sqrt(x[i][i] * x[j][j])
    return x
 ```

## Covariance Matrix
```python
def covariance_matrix(X):
    n = len(X)
    return 1 / (n - 1) * dot(de_mean_matrix(X), de_mean_matrix(X).T)
```

## Correlation Matrix
```python
def correlation_matrix(X):
    stdev_x = sqrt(variance_matrix(X))
    stdev_y = sqrt(variance_matrix(X))
    if stdev_x > 0 and stdev_y > 0:
        return covariance_matrix(X) / stdev_x / stdev_y
    else:
        return 0
```

## Variance Matrix
```python
def variance_matrix(X):
    n = len(X)
    return 1 / (n - 1) * dot(de_mean_matrix(X), de_mean_matrix(X).T)
```

## De-mean Matrix
```python
def de_mean_matrix(X):
    mean_X = mean_matrix(X)
    return [de_mean(x_i, mean_X) for x_i in X]
```

## Mean Matrix
```python
def mean_matrix(X):
    return [mean(x_i) for x_i in X]
```

## De-mean
```python
def de_mean(x, mean_x):
    return [x_i - mean_x for x_i in x]
```
## Dot
```python
def dot(v, w):
    return sum(v_i * w_i for v_i, w_i in zip(v, w))
```
