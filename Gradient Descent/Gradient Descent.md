# Gradient Descent 梯度下降


## 1 feature with m examples
```python
# Function to compute gradient
def compute_gradient(x, y, w, b):
    m = x.shape[0] #数组x的第一个纬度大小 = 样本数量
    dj_dw = 0 #偏导partial derivative
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db
```


# ![alt text](formula%20derivation.jpg)

## n feature with m examples

```python
def compute_gradient(X, y, w, b): 
    m, n = X.shape # m examples, n features

    for i in range(m): #对于每个例子m
        error = (np.dot(X[i], w) + b) - y[i]
        for j in range(n): #对于每个特征n
            dj_dw += dj_dw + error * X[i, j]
        dj_db += dj_db + error
        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw
```

## compare between 1 feature & n feature

![alt text](<comparison between 1 and n feature.jpg>)