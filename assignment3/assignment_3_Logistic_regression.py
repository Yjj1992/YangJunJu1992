## Logistic regression
###############################
import numpy as np
import random

def sigmoid(z):
    return 1/(1+np.exp(-z))

def eval_loss(w, b, x_array, gt_y_array):
    avg_loss = ((-gt_y_array@(np.log(sigmoid(w*x_array+b)).T))+(-(1-gt_y_array)@(np.log(1-sigmoid(w*x_array+b)).T)))/x_array.shape[0]
    return avg_loss


def cal_step_gradient(batch_x_array, batch_gt_y_array, w, b, lr):
    avg_dw, avg_db = 0, 0
    avg_dw=(sigmoid(w*batch_x_array+b)-batch_gt_y_array)@(batch_x_array.T)/batch_x_array.shape[0]
    avg_db=sum(sigmoid(w*batch_x_array+b)-batch_gt_y_array)
    w -= lr * avg_dw
    b -= lr * avg_db
    return w, b

def train(x_array, gt_y_array, batch_size, lr, max_iter):
    w = 0
    b = 0
    num_samples = x_array.shape[0]
    for i in range(max_iter):
        batch_idxs = np.random.choice(num_samples, batch_size)
        batch_x = np.array([x_array[j] for j in batch_idxs])
        batch_y = np.array([gt_y_array[j] for j in batch_idxs])
        w, b = cal_step_gradient(batch_x, batch_y, w, b, lr)
        print('w:{0}, b:{1}'.format(w, b))
        print('loss is {0}'.format(eval_loss(w, b, x_array, gt_y_array)))

def gen_sample_data():
    w = random.randint(0, 10) + random.random()		# for noise random.random[0, 1)
    b = random.randint(0, 5) + random.random()
    num_samples = 100
    x_list = []
    y_list = []
    for i in range(num_samples):
        x = random.randint(0, 100) * random.random()
        y = w * x + b + random.random() * random.randint(-1, 1)
        x_list.append(x)
        y_list.append(y)
    x_array=np.array(x_list)
    y_array=np.array(y_list)
    #逻辑回归因为使用了sigmoid函数，故是与0和1进行比较，如果还使用线性回归的随机数据生成方法，将会使得loss很大，故需要随机进行标签的真实数据
    y_array[y_array<=50]=0
    y_array[y_array>50]=1
    return x_array, y_array, w, b

def run():
    x_array, y_array, w, b = gen_sample_data()
    lr = 0.001
    max_iter = 10000
    train(x_array, y_array, 50, lr, max_iter)

if __name__ == '__main__':	# 跑.py的时候，跑main下面的；被导入当模块时，main下面不跑，其他当函数调
    run()
