# Run this cell to import packages.
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from gensim.models import KeyedVectors
from utils import get_vectors

'''
通常在NLP任务中，各单词会用词向量的形式来表示，词向量能对词的含义进行编码。
词向量可通过多种不同的机器学习方法进行训练得到。
在真实应用情况下，往往是直接使用已经训练好的词向量，并不会亲自训练。
'''

# 读取文件
data = pd.read_csv('capitals.txt', delimiter=' ')
print(type(data)) # <class 'pandas.core.frame.DataFrame'>
# 设置DataFrame的4个列名
data.columns = ['city1', 'country1', 'city2', 'country2']

# print first five elements in the DataFrame
# print(data.head(5))

# pickle是 Python 中用于对象序列化和反序列化的模块。使用pickle加载文件，以二进制读取模式（"rb"）打开。
word_embeddings = pickle.load(open("word_embeddings_subset.p", "rb"))
print(len(word_embeddings))  # 243

# 每一个词嵌入都是一个300维的向量
print("dimension: {}".format(word_embeddings['Spain'].shape[0])) # 300，word_embeddings[country]是(300, )的数组


def cosine_similarity(A, B):
    '''
    求向量A，B的余弦相似度

    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        cos: numerical number representing the cosine similarity between A and B.
    '''

    # 点乘
    dot = np.dot(A, B)
    # A的模
    norma = np.sqrt(np.sum(np.square(A)))
    normb = np.sqrt(np.sum(np.square(B)))
    cos = dot / (norma * normb)

    return cos

def euclidean(A, B):
    """
    求欧式距离
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        d: numerical number representing the Euclidean distance between A and B.
    """
    d = np.sqrt(np.sum(np.square(A - B)))

    return d


def get_country(city1, country1, city2, embeddings):
    """
    输入三个单词，根据第一第二个单词之间的联系，来获得第三个单词对应的国家。embeddings是词嵌入字典
    Input:
        city1: a string (the capital city of country1)
        country1: a string (the country of capital1)
        city2: a string (the capital city of country2)
        embeddings: a dictionary where the keys are words and values are their embeddings
    Output:
        countries: a dictionary with the most likely country and its similarity score
    """

    # store the city1, country 1, and city 2 in a set called group
    group = set((city1, country1, city2))

    # 获取三个地点的词嵌入量
    city1_emb = embeddings[city1]

    country1_emb = embeddings[country1]

    city2_emb = embeddings[city2]

    # get embedding of country 2 (it's a combination of the embeddings of country 1, city 1 and city 2)
    # Remember: King - Man + Woman = Queen
    vec = country1_emb - city1_emb + city2_emb

    # 初始化相似度
    similarity = -1

    # 初始化国家
    country = ''

    # loop through all words in the embeddings dictionary
    for word in embeddings.keys():

        if word not in group:

            # 获取当前word的嵌入量
            word_emb = embeddings[word]

            # 计算country2与该word在词嵌入字典中的余弦相似度
            cur_similarity = cosine_similarity(vec, word_emb)

            if cur_similarity > similarity:
                similarity = cur_similarity

                country = (word, similarity)

    return country


def get_accuracy(word_embeddings, data):
    '''
    计算模型在数据集上的准确度
    Input:
        word_embeddings: a dictionary where the key is a word and the value is its embedding
        data: a pandas dataframe containing all the country and capital city pairs

    Output:
        accuracy: the accuracy of the model
    '''

    # 初始化正确数
    num_correct = 0

    # 遍历数据，data.iterrows() 是一个用于遍历 DataFrame 对象行的生成器方法。
    for i, row in data.iterrows():

        # 获取城市与国家
        city1 = row['city1']

        country1 = row['country1']

        city2 = row['city2']

        country2 = row['country2']

        # 预测country2并与正确值进行比较
        predicted_country2, _ = get_country(city1, country1, city2, word_embeddings)

        if predicted_country2 == country2:
            num_correct += 1

    m = len(data)

    accuracy = num_correct / m

    return accuracy

accuracy = get_accuracy(word_embeddings, data)
print(f"Accuracy is {accuracy:.2f}")

'''
  ------------------------------------------------------------
  在本实验中，使用了300维的向量空间，虽然从计算的角度看该方法表现很好，
  但是300维的数据并不能直观进行可视化和理解，因此需要使用PCA进行降维。
  PCA方法通过在保持最大信息的前提下，将高维向量投影至低维空间中。
  最大信息指，原始向量和投影后的向量之间的欧式距离最小，
  因此在高维空间中彼此接近的向量降维后仍彼此接近。
 ------------------------------------------------------------
'''


def compute_pca(X, n_components=2):
    """
    计算主成分分析
    Input:
        X: 维度为 (m,n)，每一行对应一个词向量。
        n_components: 想要保存的维度
    Output:
        X_reduced: 数据转换为 2 维 / 列 + 重新生成原始数据。
    """

    # 对数据进行均值归一化，这一步确保不同特征对 PCA 的贡献是平等的。
    X_demeaned = X - X.mean(axis=0, keepdims=True)

    # 计算协方差矩阵。协方差矩阵的每个元素 Cov(Xi, Xj)表示特征Xi和特征Xj之间的线性相关性。
    # rowvar=False表示将每一列看作一个变量，即列是变量，行是观测值。
    # 如果不设置这个参数为False，默认情况下np.cov函数会将每一行看作一个变量，即行是变量，列是观测值。
    covariance_matrix = np.cov(X_demeaned, rowvar=False)

    # np.linalg.eigh 用于计算一个实对称矩阵或厄米特矩阵的特征值和特征向量。
    # eigen_vals 是一个一维数组，包含了输入矩阵的特征值，默认按照升序排列。
    # eigen_vecs 是一个二维数组，其中每一列是对应特征值的特征向量。
    eigen_vals, eigen_vecs = np.linalg.eigh(covariance_matrix)

    # 返回特征值从小到大的索引。这样可以按顺序重新排列特征值和对应的特征向量。
    idx_sorted = np.argsort(eigen_vals)

    # 将特征值的索引反转，变为降序排列。PCA 只保留方差最大的方向，因此需要从大到小选择特征值。
    idx_sorted_decreasing = idx_sorted[::-1]

    # 使用降序排列的索引来重排特征值。
    eigen_vals_sorted = eigen_vals[idx_sorted_decreasing]

    # 使用降序排列的索引来重排特征向量，以匹配降序排列的特征值。
    eigen_vecs_sorted = eigen_vecs[:,idx_sorted_decreasing]

    # 选择前  n_components 个特征向量（对应于最大的特征值），这些特征向量代表数据的主成分方向。
    eigen_vecs_subset = eigen_vecs_sorted[:,:n_components]

    # 将数据投影到选取的主成分方向上。通过矩阵乘法实现降维操作。这里是将原始数据矩阵  X  投影到低维空间的过程。
    X_reduced = np.matmul(eigen_vecs_subset.T,X_demeaned.T).T

    return X_reduced

# 打印降维后的矩阵
np.random.seed(1)
X = np.random.rand(3, 10)
X_reduced = compute_pca(X, n_components=2)
print("Your original matrix was " + str(X.shape) + " and it became:")
print(X_reduced)


# 使用上面实现的PCA函数来进行可视化。
words = ['oil', 'gas', 'happy', 'sad', 'city', 'town',
         'village', 'country', 'continent', 'petroleum', 'joyful']

# 给定一个单词列表和嵌入向量，它返回一个包含所有嵌入向量的矩阵。
X = get_vectors(word_embeddings, words)

result = compute_pca(X, 2)
plt.scatter(result[:, 0], result[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0] - 0.05, result[i, 1] + 0.1))

plt.show()