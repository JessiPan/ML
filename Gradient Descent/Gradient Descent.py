import math, copy
import numpy as np #scientific computing
import matplotlib.pyplot as plt #绘图plotting data
plt.style.use('./deeplearning.mplstyle')
from lab_utils_uni import plt_house_x,plt_contour_wgrad,plt_divergence, plt_gradients

# Load dataset
x_train = np.array([1.0, 2.0]) #feature特征/input
y_train = np.array([300.0, 500.0]) #target value/label

# Function to calculate the cost
def compute_cost(x, y, w, b):

    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2 #squared error均方误差
    total_cost = 1 / (2 * m) * cost

    return total_cost

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

plt_gradients(x_train, y_train, compute_cost, compute_gradient)
plt.show()

# Function to implement gradient_descent
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    w = copy.deepcopy(w_in) #保持纯函数特性：1、仅依赖输入参数（同输入输出 ）；2、不会修改外部状态或变量（如全局变量）
    J_history = [] #  History of cost value [Jw,b(x)]
    p_history = [] # History of parameters [w,b]
    b = b_in
    w = w_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b) # calculate & update the partial derivative term
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000: # prevent resource exhaustion
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w,b])

        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}, Cost {J_history[-1]:0.2e}",
                  f"dj_dw: {dj_dw:0.3e}, dj_db: {dj_db:0.3e}",
                  f"w:{w:0.3e}, b:{b:0.5e}")
            #f-string格式化字符打印信息：i:4（4位数），0.3e（保留3位小数）
        return w, b, J_history, p_history
#initialize parameter
w_init = 0
b_init = 0
#gradient descent setting
iterations = 10000
tmp_alpha = 1.0e-2

w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)

print(f"(w,b) found by gradient descent: ({w_final: 8.4f}, {b_final:8.4f})") #format格式

#plot cost vs iteration
fig, (ax1, ax2) = plt.subplot(1, 2, constrained_layout = True, figsize=(12,4)) # 1 行 2 列的子图布局，inch单位
ax1.plot(J_hist[:100]) #前100个元素:0-99【slicing切片操作符】
ax2.plot(1000 + np.arange(len((J_hist[1000:])), J_hist[1000:])) #x轴【因为默认从1开始】【1000-末尾；np.arange生成等差数列】，y轴【】
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step')
plt.show()

#predict with optimal values
print(f"1000 sqft house prediction {w_final*1.0 + b_final:0.1f} Thousand dollars")
print(f"1200 sqft house prediction {w_final*1.2 + b_final:0.1f} Thousand dollars")
print(f"2000 sqft house prediction {w_final*2.0 + b_final:0.1f} Thousand dollars")

#contour
fig, ax = plt.subplots(1,1, figsize=(12, 6))
plt_contour_wgrad(x_train, y_train, p_hist, ax)