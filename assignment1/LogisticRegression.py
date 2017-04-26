import autodiff as ad
import numpy as np

# construct the computation graph
x = ad.Variable(name="x")
w = ad.Variable(name="w")
y_ = ad.Variable(name="lables")

prob = 1.0/(1.0 + ad.exp_op((-1.0 * ad.matmul_op(w, x))))
loss = -1.0 * ad.reduce_sum_op(y_ * ad.log_op(prob) + (1.0 - y_) * ad.log_op(1.0 - prob), axis=1)

# pay attention that there is a ','
grad_w, = ad.gradients(loss, [w])

# Data
data1 = np.random.normal(1, 0.1, size=(100, 10))
data2 = np.random.normal(5, 0.4, size=(200, 10))
data = np.concatenate((data1, data2), axis=0).T
x_val = np.concatenate((data, np.ones((1, 300))), axis=0)
y_val = np.concatenate((np.zeros((data1.shape[0], 1)), np.ones((data2.shape[0], 1))), axis=0).T
# Variables
w_val = np.random.normal(size=(1, 11))

# Params
learning_rate = 0.0001

# Execute
executor = ad.Executor([loss, grad_w])

for i in xrange(100000):
    # evaluate the graph
    loss_val, grad_w_val = executor.run(feed_dict={x: x_val, w: w_val, y_: y_val})
    # update the parameters using SGD
    w_val -= learning_rate * grad_w_val

    if i % 1000 == 0:
        print loss_val



