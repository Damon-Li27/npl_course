import nltk
import numpy as np
from utils2 import get_batches, get_dict, compute_pca
import tensorflow as tf
import matplotlib.pyplot as plt
import re

'''
    1、数据预处理
'''
nltk.download("punkt_tab")

with open('shakespeare.txt') as f:
    data = f.read()
#  使用.替换标点符号
data = re.sub(r'[,!?;-]', '.', data)
#  文本分割为单词
data = nltk.word_tokenize(data)
#  转为小写并去除非字母符号，此时data为一个词汇表
data = [ch.lower() for ch in data if ch.isalpha() or ch == '.']

# 添加单词与索引之间的映射
word2Ind, Ind2word = get_dict(data)
V = len(word2Ind)

'''
    2、训练CBOW模型
'''
def initialize_model(N, V, random_seed=1):
    '''
    Inputs:
        N:  隐藏层的隐藏单元数
        V:  词汇表个数
        random_seed: random seed for consistent results in the unit tests
     Outputs:
        W1, W2, b1, b2: initialized weights and biases
    '''

    np.random.seed(random_seed)

    # 根据维度初始化参数
    W1 = tf.Variable(tf.keras.initializers.GlorotNormal(seed=1)(shape=(N, V)))

    W2 = tf.Variable(tf.keras.initializers.GlorotNormal(seed=1)(shape=(V, N)))

    b1 = tf.Variable(tf.keras.initializers.GlorotNormal(seed=1)(shape=(N, 1)))

    b2 = tf.Variable(tf.keras.initializers.GlorotNormal(seed=1)(shape=(V, 1)))

    return W1, W2, b1, b2


def forward_prop(x, W1, W2, b1, b2):
    '''
    前向传播
    '''

    z1 = tf.add(tf.matmul(W1, x), b1)

    a1 = tf.nn.relu(z1)

    z2 = tf.add(tf.matmul(W2, a1), b2)

    return z2


def compute_cost(y, z):
    logits = tf.transpose(z)
    labels = tf.transpose(y)
    cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cost = tf.reduce_mean(cost)
    return cost


def model(N, V, num_iters, learning_rate=0.003):
    C = 2
    batch_size = 128
    iters = 0
    W1, W2, b1, b2 = initialize_model(N, V)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)


    # y对应中心词索引的位置设为 1，其余全部为0（独热编码），长度为V
    # x对于上下文的向量表示。其索引位置的值为 “在上下文中出现的次数/上下文词的总数”，其余为0
    for x, y in get_batches(data, word2Ind, V, C, batch_size):
        with tf.GradientTape() as tape:
            z2 = forward_prop(x, W1, W2, b1, b2)
            cost = compute_cost(y, z2)

        trainable_variables = [W1, W2, b1, b2]
        grads = tape.gradient(cost, trainable_variables)
        optimizer.apply_gradients(zip(grads, trainable_variables))
        if (iters + 1) % 10 == 0:
            print(f"iters: {iters + 1} cost: {cost:.6f}")
        iters += 1

        if iters == num_iters:
            break
    return W1, W2

N = 50
word2Ind, Ind2word = get_dict(data)
V = len(word2Ind)
W1, W2 = model(N, V, 200)

words = ['king', 'queen','lord','man', 'woman','dog','wolf',
         'rich','happy','sad']

embs = tf.divide(tf.add(tf.transpose(W1), W2), 2.0)

# 对于给定的单词，返回其嵌入量矩阵
idx = [word2Ind[word] for word in words]
X = tf.gather(embs, idx)
print(X.shape, idx)  # X.shape:  Number of words of dimension N each

result = compute_pca(X.numpy(), 2)
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.show()


result= compute_pca(X.numpy(), 4)
plt.scatter(result[:, 3], result[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 3], result[i, 1]))
plt.show()
