import nltk

import pickle
import numpy as np
from nltk.corpus import twitter_samples

from utils import (cosine_similarity, get_dict,
                   process_tweet)

nltk.download('stopwords')
nltk.download('twitter_samples')

# en_embeddings_subset是英文单词对应的嵌入量的字典，key为word,value为嵌入量的set
en_embeddings_subset = pickle.load(open("en_embeddings.p", "rb"))
# fr_embeddings_subset是法语单词对应的嵌入量
fr_embeddings_subset = pickle.load(open("fr_embeddings.p", "rb"))

'''
加载英语对法语的字典
`en_fr_train` 是一个key为英文，value为英文对应的法语的字典，比如

{'the': 'la',
 'and': 'et',
 'was': 'était',
 'for': 'pour',
 ...
 
 en-fr.test是同样的测试集
'''
en_fr_train = get_dict('en-fr.train.txt')
print('The length of the English to French training dictionary is', len(en_fr_train))
en_fr_test = get_dict('en-fr.test.txt')
print('The length of the English to French test dictionary is', len(en_fr_train))


def get_matrices(en_fr, french_vecs, english_vecs):
    """
    由XR=Y，求转换矩阵R，需要先获取两个矩阵，即英文词向量的X矩阵和对应法文词向量的Y矩阵。
    方法：如果french_vecs中的法语单词和english_vecs中的英语单词，在映射字典en_fr中都存在，就获取其嵌入量并存为矩阵
    Input:
        en_fr: 英语到法语的字典
        french_vecs: 法语单词嵌入量
        english_vecs: 英文单词嵌入量
    Output:
        X: 列是英语嵌入量的矩阵
        Y: 列是对应的法语嵌入量的矩阵
        R: 使 F 范数最小化的投影矩阵。
    """

    # X_l and Y_l are lists of the English and French word embeddings
    X_l = list()
    Y_l = list()

    # 获取所有英文单词和法文单词
    english_set = english_vecs.keys()
    french_set = french_vecs.keys()

    # 获取en_fr字典中的法文单词，存储到french_words
    french_words = set(en_fr.values())

    # 遍历英文-法文字典中的所有key和value
    for en_word, fr_word in en_fr.items():

        # 如果字典中的单词拥有对应的嵌入量，就将其嵌入量存起来
        if fr_word in french_set and en_word in english_set:
            # get the english embedding
            en_vec = english_vecs[en_word]

            # get the french embedding
            fr_vec = french_vecs[fr_word]

            # add the english embedding to the list
            X_l.append(en_vec)

            # add the french embedding to the list
            Y_l.append(fr_vec)

    # 将列表X_l和Y_l转为矩阵，比如：
    # X_l = [[1, 2, 3], [4, 5, 6]]
    # X = np.vstack(X_l)
    # print(X)
    # array([[1, 2, 3],
    #        [4, 5, 6]])
    X = np.vstack(X_l)

    Y = np.vstack(Y_l)

    return X, Y


X_train, Y_train = get_matrices(
    en_fr_train, fr_embeddings_subset, en_embeddings_subset)


def compute_loss(X, Y, R):
    '''
    计算损失函数
    Inputs:
        X: a matrix of dimension (m,n) where the columns are the English embeddings.
        Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
        R: a matrix of dimension (n,n) - transformation matrix from English to French vector space embeddings.
    Outputs:
        L: a matrix of dimension (m,n) - the value of the loss function for given X, Y and R.
    '''
    m = X.shape[0]

    # 差值
    diff = np.dot(X, R) - Y
    # 损失值
    loss = np.sum(np.square(diff)) / m
    return loss


def compute_gradient(X, Y, R):
    '''
    计算梯度
    Inputs:
        X: a matrix of dimension (m,n) where the columns are the English embeddings.
        Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
        R: a matrix of dimension (n,n) - transformation matrix from English to French vector space embeddings.
    Outputs:
        g: a matrix of dimension (n,n) - gradient of the loss function L for given X, Y and R.
    '''
    # m is the number of rows in X
    m = X.shape[0]

    # gradient is X^T(XR - Y) * 2/m
    gradient = np.dot(X.T, np.dot(X, R) - Y) * 2 / m
    return gradient


def align_embeddings(X, Y, train_steps=100, learning_rate=0.0003):
    '''
    训练R
    Inputs:
        X: a matrix of dimension (m,n) where the columns are the English embeddings.
        Y: a matrix of dimension (m,n) where the columns correspong to the French embeddings.
        train_steps: positive int - describes how many steps will gradient descent algorithm do.
        learning_rate: positive float - describes how big steps will  gradient descent algorithm do.
    Outputs:
        R: a matrix of dimension (n,n) - the projection matrix that minimizes the F norm ||X R -Y||^2
    '''
    np.random.seed(129)

    # the number of columns in X is the number of dimensions for a word vector (e.g. 300)
    # R is a square matrix with length equal to the number of dimensions in th  word embedding
    R = np.random.rand(X.shape[1], X.shape[1])

    for i in range(train_steps):
        loss = compute_loss(X, Y, R)
        if i % 25 == 0:
            print(f"loss at iteration {i} is: {loss:.4f}")
        dR = compute_gradient(X, Y, R)

        R -= learning_rate * dR
    return R

# R_train = align_embeddings(X_train, Y_train, train_steps=400, learning_rate=0.8)

'''
    以上代码已经获得了转换矩阵R，使得XR≈Y。
'''
def nearest_neighbor(v, candidates, k=1):
    """
    v为原向量（1个），candidates为v对应的候选向量（多个），目的是从candidates中选出前k个最接近的目标项目
    Input:
      - v, 将要为其找到临近数据的向量
      - candidates: 找到的临近数据的向量
      - k: 前k的最临近
    Output:
      - k_idx: 前 k 个最接近向量的索引。
    """
    similarity_l = []

    for row in candidates:
        # 余弦相似度
        cos_similarity = np.dot(v, row) / (np.linalg.norm(v) * np.linalg.norm(row))

        similarity_l.append(cos_similarity)

    # 获取排序索引，argsort默认从小到大排，使用[::-1]反转
    sorted_ids = np.argsort(similarity_l)[::-1]

    k_idx = sorted_ids[:k]
    return k_idx

def test_vocabulary(X, Y, R):
    '''
    测试翻译效果以及准确度
    Input:
        X: a matrix where the columns are the English embeddings.
        Y: a matrix where the columns correspond to the French embeddings.
        R: the transform matrix which translates word embeddings from
        English to French word vector space.
    Output:
        accuracy: for the English to French capitals
    '''

    # 预测值：XR
    pred = np.dot(X, R)

    num_correct = 0

    for i in range(len(pred)):
        # get the index of the nearest neighbor of pred at row 'i'; also pass in the candidates in Y
        pred_idx = nearest_neighbor(pred[i], Y, 1)

        # 验证集中X词变量与Y的词变量是一一对应的，XR矩阵的形状与X形状相同。如果XR所求得的pred的第i列，它的最临近变量正好是Y的第i列，那就是对的。
        if pred_idx == i:
            # increment the number correct by 1.
            num_correct += 1

    accuracy = num_correct / len(Y)

    return accuracy

# 获取测试集的词向量
X_val, Y_val = get_matrices(en_fr_test, fr_embeddings_subset, en_embeddings_subset)

# acc = test_vocabulary(X_val, Y_val, R_train)
# print(f"accuracy on test set is {acc:.3f}") # 0.557

'''
    以上实现了得到转换矩阵R，通过XR变换得到预测值pred，将pred前k个最接近的结果与Y比较余弦相似度，
    得到最接近的那个结果的索引，看该索引是否还是与X对应的Y的索引。并测试了验证集的准确度。
    下面使用局部敏感哈希算法。
'''

all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')
all_tweets = all_positive_tweets + all_negative_tweets

def get_document_embedding(tweet, en_embeddings):
    '''
    获取推文的英文词向量
    Input:
        - tweet: a string
        - en_embeddings: a dictionary of word embeddings
    Output:
        - doc_embedding: sum of all word embeddings in the tweet
    '''
    doc_embedding = np.zeros(300)

    # 处理推文为单词列表
    processed_doc = process_tweet(tweet)
    for word in processed_doc:
        # 将单个单词的词向量全部相加
        doc_embedding += en_embeddings.get(word, 0)
    return doc_embedding

def get_document_vecs(all_docs, en_embeddings):
    '''
    Input:
        - all_docs: list of strings - all tweets in our dataset.
        - en_embeddings: dictionary with words as the keys and their embeddings as the values.
    Output:
        - document_vec_matrix: matrix of tweet embeddings.
        - ind2Doc_dict: dictionary with indices of tweets in vecs as keys and their embeddings as the values.
    '''

    # the dictionary's key is an index (integer) that identifies a specific tweet
    # the value is the document embedding for that document
    ind2Doc_dict = {}

    # this is list that will store the document vectors
    document_vec_l = []

    for i, doc in enumerate(all_docs):

        # 获取推文的文档向量
        doc_embedding = get_document_embedding(doc, en_embeddings)

        # save the document embedding into the ind2Tweet dictionary at index i
        ind2Doc_dict[i] = doc_embedding

        # append the document embedding to the list of document vectors
        document_vec_l.append(doc_embedding)

    # convert the list of document vectors into a 2D array (each row is a document vector)
    document_vec_matrix = np.vstack(document_vec_l)

    return document_vec_matrix, ind2Doc_dict


document_vecs, ind2Tweet = get_document_vecs(all_tweets, en_embeddings_subset)

'''
    LSH实现
'''
N_VECS = len(all_tweets)      # 向量的数量 (推文向量的总数)
N_DIMS = len(ind2Tweet[1])    # 向量的维度 (推文向量的维度, 例如300维)
print(f"Number of vectors is {N_VECS} and each has {N_DIMS} dimensions.")
# 超平面数
N_PLANES = 10
# 重复hash计算的次数以改进搜索。定义哈希计算的重复次数，这相当于生成多个不同的哈希函数集合来增加搜索的鲁棒性。
N_UNIVERSES = 25


np.random.seed(0)
# 创建了 `N_UNIVERSES` 个超平面，每个超平面用一个形状为 `(N_DIMS, N_PLANES)` 的矩阵来表示，表示超平面的方向。这些超平面是随机生成的。
planes_l = [np.random.normal(size=(N_DIMS, N_PLANES))
            for _ in range(N_UNIVERSES)]


def hash_value_of_vector(v, plane):
    """
    用于计算一个向量 `v` 投影在给定超平面 `plane` 上的哈希值。
    基本思路是计算向量在多个超平面上的投影，确定向量位于超平面的哪一侧，并根据这些信息生成哈希值。
    Input:
        - v:  推文的向量，维度为 (1, N_DIMS)
        - planes: 矩阵，维度为 (N_DIMS, N_PLANES) - 划分区域的平面集合。
    Output:
        - res: 向量hash值
    """
    # 计算向量与超平面之间的点积。点积的符号决定向量是在超平面的一侧还是另一侧。
    dot_product = np.dot(v, plane)

    # sign是一个函数，用于返回数组中每个元素的符号。对于正数，返回 1。对于零，返回 0。对于负数，返回 -1。
    sign_of_dot_product = np.sign(dot_product)

    # 这里将符号数组转换为布尔值：正数（或0）被视为 `True`，负数为 `False`。
    h = sign_of_dot_product >= 0

    h = np.squeeze(h)

    # 初始化hash值
    hash_value = 0

    n_planes = plane.shape[1]
    for i in range(n_planes):
        # 通过位移运算生成哈希值，h[i] 为布尔值，将其乘以 2^i 以生成唯一哈希
        hash_value += 2 ** i * h[i]

    hash_value = int(hash_value)

    return hash_value


def make_hash_table(vecs, planes):
    """
    创建hash表，将输入的向量映射到基于随机超平面的hash桶
    Input:
        - vecs: 需要哈希的向量列表
        - planes: 超平面矩阵，用于生成hash值，维度为(向量为度，平面数量)
        - hash_table: dictionary - 哈希表字典，键是hash值，值是向量列表
        - id_table: dictionary - 哈希表索引，键是hash值，值是向量索引列表
    """
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

    # number of planes is the number of columns in the planes matrix
    num_of_planes = planes.shape[1]

    # number of buckets is 2^(number of planes)
    num_buckets = 2**num_of_planes

    # create the hash table as a dictionary.
    # Keys are integers (0,1,2.. number of buckets)
    # Values are empty lists
    hash_table = {i:[] for i in range(0,num_buckets)}

    # create the id table as a dictionary.
    # Keys are integers (0,1,2... number of buckets)
    # Values are empty lists
    id_table = {i:[] for i in range(0,num_buckets)}

    # for each vector in 'vecs'
    for i, v in enumerate(vecs):
        # calculate the hash value for the vector
        h = hash_value_of_vector(v,planes)

        # store the vector into hash_table at key h,
        # by appending the vector v to the list at key h
        hash_table[h].append(v)

        # store the vector's index 'i' (each document is given a unique integer 0,1,2...)
        # the key is the h, and the 'i' is appended to the list at key h
        id_table[h].append(i)

    ### END CODE HERE ###
    return hash_table, id_table

# 创建所有hash表
hash_tables = []
id_tables = []
for universe_id in range(N_UNIVERSES):  # there are 25 hashes
    print('working on hash universe #:', universe_id)
    planes = planes_l[universe_id]
    hash_table, id_table = make_hash_table(document_vecs, planes)
    hash_tables.append(hash_table)
    id_tables.append(id_table)


def approximate_knn(doc_id, v, planes_l, k=1, num_universes_to_use=N_UNIVERSES):
    """执行近似最近邻搜索，返回与查询向量最相似的文档列表"""
    assert num_universes_to_use <= N_UNIVERSES

    # Vectors that will be checked as possible nearest neighbor
    vecs_to_consider_l = list()

    # list of document IDs
    ids_to_consider_l = list()

    # create a set for ids to consider, for faster checking if a document ID already exists in the set
    ids_to_consider_set = set()

    # loop through the universes of planes
    for universe_id in range(num_universes_to_use):

        # get the set of planes from the planes_l list, for this particular universe_id
        planes = planes_l[universe_id]

        # get the hash value of the vector for this set of planes
        hash_value = hash_value_of_vector(v, planes)

        # get the hash table for this particular universe_id
        hash_table = hash_tables[universe_id]

        # get the list of document vectors for this hash table, where the key is the hash_value
        document_vectors_l = hash_table[hash_value]

        # get the id_table for this particular universe_id
        id_table = id_tables[universe_id]

        # get the subset of documents to consider as nearest neighbors from this id_table dictionary
        new_ids_to_consider = id_table[hash_value]

        ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

        # remove the id of the document that we're searching
        if doc_id in new_ids_to_consider:
            new_ids_to_consider.remove(doc_id)
            print(f"removed doc_id {doc_id} of input vector from new_ids_to_search")

        # loop through the subset of document vectors to consider
        for i, new_id in enumerate(new_ids_to_consider):

            # if the document ID is not yet in the set ids_to_consider...
            if new_id not in ids_to_consider_set:
                # access document_vectors_l list at index i to get the embedding
                # then append it to the list of vectors to consider as possible nearest neighbors
                vecs_to_consider_l.append(document_vectors_l[i])

                # append the new_id (the index for the document) to the list of ids to consider
                ids_to_consider_l.append(new_id)

                # also add the new_id to the set of ids to consider
                # (use this to check if new_id is not already in the IDs to consider)
                ids_to_consider_set.add(new_id)

        ### END CODE HERE ###

    # Now run k-NN on the smaller set of vecs-to-consider.
    print("Fast considering %d vecs" % len(vecs_to_consider_l))

    # convert the vecs to consider set to a list, then to a numpy array
    vecs_to_consider_arr = np.array(vecs_to_consider_l)

    # call nearest neighbors on the reduced list of candidate vectors
    nearest_neighbor_idx_l = nearest_neighbor(v, vecs_to_consider_arr, k=k)

    # Use the nearest neighbor index list as indices into the ids to consider
    # create a list of nearest neighbors by the document ids
    nearest_neighbor_ids = [ids_to_consider_l[idx]
                            for idx in nearest_neighbor_idx_l]

    return nearest_neighbor_ids


doc_id = 0
doc_to_search = all_tweets[doc_id]
vec_to_search = document_vecs[doc_id]
#%%
# UNQ_C22 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# You do not have to input any code in this cell, but it is relevant to grading, so please do not change anything

# Sample
nearest_neighbor_ids = approximate_knn(
    doc_id, vec_to_search, planes_l, k=3, num_universes_to_use=5)