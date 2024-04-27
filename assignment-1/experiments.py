
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
import timeit
import matplotlib.pyplot as plt

np.random.seed(42)
num_average_time = 100

# Learn DTs 
# ...
# 
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
# ..
# Function to create fake data (take inspiration from usage.py)
# ...
# ..other functions

P = 4
N = 100
N_test = np.arange(30,100,5)
P_test = np.arange(4,20,2)

tree = DecisionTree(criterion='information_gain', max_depth=10)


# # Real Input Real Output

# train_n = []
# predict_n = []
# train_p = []
# predict_p = []

# for n in N_test:

#     X = pd.DataFrame(np.random.randn(n, P))
#     y = pd.Series(np.random.randn(n))

#     start = timeit.default_timer()
#     tree.fit(X, y)
#     mid = timeit.default_timer()
#     y_hat = tree.predict(X)
#     end = timeit.default_timer()

#     train_n.append(mid-start)
#     predict_n.append(end-mid)

# for p in P_test:

#     X = pd.DataFrame(np.random.randn(N, p))
#     y = pd.Series(np.random.randn(N))

#     start = timeit.default_timer()
#     tree.fit(X, y)
#     mid = timeit.default_timer()
#     y_hat = tree.predict(X)
#     end = timeit.default_timer()

#     train_p.append(mid-start)
#     predict_p.append(end-mid)

# fig1 = plt.figure()

# plt.subplot(2,2,1)
# plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
# plt.plot(N_test, train_n)
# plt.xlabel('Number of Samples')
# plt.ylabel('Time')
# plt.title('Train time with varying N')

# plt.subplot(2,2,2)
# plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
# plt.plot(P_test, train_p)
# plt.xlabel('Number of Features')
# plt.ylabel('Time')
# plt.title('Train time with varying P')

# plt.subplot(2,2,3)
# plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
# plt.plot(N_test, predict_n)
# plt.xlabel('Number of Samples')
# plt.ylabel('Time')
# plt.title('Predict time with varying N')

# plt.subplot(2,2,4)
# plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
# plt.plot(P_test, predict_p)
# plt.xlabel('Number of Features')
# plt.ylabel('Time')
# plt.title('Predict time with varying P')

# plt.suptitle('Real Input Real Output')
# plt.savefig('q4_riro.png')
# plt.show()




# #Real Input Discrete Output

# train_n = []
# predict_n = []
# train_p = []
# predict_p = []

# for n in N_test:

#     X = pd.DataFrame(np.random.randn(n, P))
#     y = pd.Series(np.random.randint(P, size = n), dtype="category")

#     start = timeit.default_timer()
#     tree.fit(X, y)
#     mid = timeit.default_timer()
#     y_hat = tree.predict(X)
#     end = timeit.default_timer()

#     train_n.append(mid-start)
#     predict_n.append(end-mid)

# for p in P_test:

#     X = pd.DataFrame(np.random.randn(N, p))
#     y = pd.Series(np.random.randint(P, size = N), dtype="category")

#     start = timeit.default_timer()
#     tree.fit(X, y)
#     mid = timeit.default_timer()
#     y_hat = tree.predict(X)
#     end = timeit.default_timer()

#     train_p.append(mid-start)
#     predict_p.append(end-mid)

# fig1 = plt.figure()

# plt.subplot(2,2,1)
# plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
# plt.plot(N_test, train_n)
# plt.xlabel('Number of Samples')
# plt.ylabel('Time')
# plt.title('Train time with varying N')

# plt.subplot(2,2,2)
# plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
# plt.plot(P_test, train_p)
# plt.xlabel('Number of Features')
# plt.ylabel('Time')
# plt.title('Train time with varying P')

# plt.subplot(2,2,3)
# plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
# plt.plot(N_test, predict_n)
# plt.xlabel('Number of Samples')
# plt.ylabel('Time')
# plt.title('Predict time with varying N')

# plt.subplot(2,2,4)
# plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
# plt.plot(P_test, predict_p)
# plt.xlabel('Number of Features')
# plt.ylabel('Time')
# plt.title('Predict time with varying P')

# plt.suptitle('Real Input Discrete Output')
# plt.savefig('q4_rido.png')
# plt.show()





# #Discrete Input Real Output

# train_n = []
# predict_n = []
# train_p = []
# predict_p = []

# for n in N_test:

#     X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = n), dtype="category") for i in range(P)})
#     y = pd.Series(np.random.randn(n))

#     start = timeit.default_timer()
#     tree.fit(X, y)
#     mid = timeit.default_timer()
#     y_hat = tree.predict(X)
#     end = timeit.default_timer()

#     train_n.append(mid-start)
#     predict_n.append(end-mid)

# for p in P_test:

#     X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in range(p)})
#     y = pd.Series(np.random.randn(N))

#     start = timeit.default_timer()
#     tree.fit(X, y)
#     mid = timeit.default_timer()
#     y_hat = tree.predict(X)
#     end = timeit.default_timer()

#     train_p.append(mid-start)
#     predict_p.append(end-mid)

# fig1 = plt.figure()

# plt.subplot(2,2,1)
# plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
# plt.plot(N_test, train_n)
# plt.xlabel('Number of Samples')
# plt.ylabel('Time')
# plt.title('Train time with varying N')

# plt.subplot(2,2,2)
# plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
# plt.plot(P_test, train_p)
# plt.xlabel('Number of Features')
# plt.ylabel('Time')
# plt.title('Train time with varying P')

# plt.subplot(2,2,3)
# plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
# plt.plot(N_test, predict_n)
# plt.xlabel('Number of Samples')
# plt.ylabel('Time')
# plt.title('Predict time with varying N')

# plt.subplot(2,2,4)
# plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
# plt.plot(P_test, predict_p)
# plt.xlabel('Number of Features')
# plt.ylabel('Time')
# plt.title('Predict time with varying P')

# plt.suptitle('Discrete Input real Output')
# plt.savefig('q4_diro.png')
# plt.show()




#Discrete Input Discrete Output

train_n = []
predict_n = []
train_p = []
predict_p = []

for n in N_test:

    X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = n), dtype="category") for i in range(P)})
    y = pd.Series(np.random.randint(P, size = n), dtype="category")


    start = timeit.default_timer()
    tree.fit(X, y)
    mid = timeit.default_timer()
    y_hat = tree.predict(X)
    end = timeit.default_timer()

    train_n.append(mid-start)
    predict_n.append(end-mid)

for p in P_test:

    X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in range(p)})
    y = pd.Series(np.random.randint(P, size = N), dtype="category")


    start = timeit.default_timer()
    tree.fit(X, y)
    mid = timeit.default_timer()
    y_hat = tree.predict(X)
    end = timeit.default_timer()

    train_p.append(mid-start)
    predict_p.append(end-mid)

fig1 = plt.figure()

plt.subplot(2,2,1)
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
plt.plot(N_test, train_n)
plt.xlabel('Number of Samples')
plt.ylabel('Time')
plt.title('Train time with varying N')

plt.subplot(2,2,2)
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
plt.plot(P_test, train_p)
plt.xlabel('Number of Features')
plt.ylabel('Time')
plt.title('Train time with varying P')

plt.subplot(2,2,3)
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
plt.plot(N_test, predict_n)
plt.xlabel('Number of Samples')
plt.ylabel('Time')
plt.title('Predict time with varying N')

plt.subplot(2,2,4)
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
plt.plot(P_test, predict_p)
plt.xlabel('Number of Features')
plt.ylabel('Time')
plt.title('Predict time with varying P')

plt.suptitle('Discrete Input discrete Output')
plt.savefig('q4_dido.png')
plt.show()