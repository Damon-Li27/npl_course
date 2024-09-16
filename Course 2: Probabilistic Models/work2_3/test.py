import math
import random
import numpy as np
import pandas as pd
import nltk


'''
    Part1：加载和处理数据集，得到预处理之后的训练集、测试集和词汇表
'''

nltk.download('punkt_tab')

with open("en_US.twitter.txt", "r") as f:
    # data为字符串类型
    data = f.read()


def split_to_sentences(data):
    """
    根据换行符\n拆分句子

    Args:
        data: str
    Returns:
        A list of sentences
    """
    sentences = data.split("\n")

    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if len(s) > 0]

    return sentences


def tokenize_sentences(sentences):
    """
    将文本拆分为一个个的标记（token），通常是单词、标点符号等

    Args:
        sentences: List of strings

    Returns:
        List of lists of tokens
    """

    tokenized_sentences = []

    for sentence in sentences:
        # 转为小写
        sentence = sentence.lower()

        # word_tokenize函数的主要作用是将输入的文本分割成一个个单独的词（token），
        # 也能正确地处理标点与单词的分离（split方法不能处理标点符号）
        # 需要先下载punkt_tab包
        tokenized = nltk.word_tokenize(sentence)

        tokenized_sentences.append(tokenized)

    return tokenized_sentences


def get_tokenized_data(data):
    """
    对以上两个函数的调用

    Args:
        data: String

    Returns:
        List of lists of tokens
    """
    sentences = split_to_sentences(data)

    tokenized_sentences = tokenize_sentences(sentences)

    return tokenized_sentences



tokenized_data = get_tokenized_data(data)
random.seed(87)
random.shuffle(tokenized_data)

train_size = int(len(tokenized_data) * 0.8)

# 训练集和测试集，均为list类型，list中每个元素是一个句子的单词list
train_data = tokenized_data[0:train_size]
test_data = tokenized_data[train_size:]


def count_words(tokenized_sentences):
    """
    计算每个单词在训练集中出现的次数（便于后面设置阈值，只需要达到指定出现次数的单词）

    Args:
        tokenized_sentences: List of lists of strings
        形如：
            tokenized_sentences = [['sky', 'is', 'blue', '.'],
                       ['leaves', 'are', 'green', '.'],
                       ['roses', 'are', 'red', '.']]
    Returns:
        dict that maps word (str) to the frequency (int)
    """

    word_counts = {}

    for sentence in (tokenized_sentences):

        for token in sentence:

            if token not in word_counts:
                word_counts[token] = 1
            else:
                word_counts[token] += word_counts[token]

    return word_counts


def get_words_with_nplus_frequency(tokenized_sentences, count_threshold):
    """
        查找出现了N次及以上的词

    Args:
        tokenized_sentences: List of lists of sentences
        count_threshold: minimum number of occurrences for a word to be in the closed vocabulary.

    Returns:
        List of words that appear N times or more
    """
    closed_vocab = []

    # 获取字典，key：word，value：count
    word_counts = count_words(tokenized_sentences)

    for word, cnt in word_counts.items():

        if cnt >= count_threshold:
            closed_vocab.append(word)

    return closed_vocab


def replace_oov_words_by_unk(tokenized_sentences, vocabulary, unknown_token="<unk>"):
    """
    使用unk来替换未出现在字典中的单词

    Args:
        tokenized_sentences: List of lists of strings
        vocabulary: List of strings that we will use
        unknown_token: A string representing unknown (out-of-vocabulary) words

    Returns:
        List of lists of strings, with words not in the vocabulary replaced
    """

    vocabulary = set(vocabulary)

    replaced_tokenized_sentences = []

    for sentence in tokenized_sentences:

        replaced_sentence = []

        for token in sentence:

            if token in vocabulary:
                replaced_sentence.append(token)
            else:
                replaced_sentence.append(unknown_token)

        replaced_tokenized_sentences.append(replaced_sentence)

    return replaced_tokenized_sentences


def preprocess_data(train_data, test_data, count_threshold):
    """
    预处理数据：
        查找出现在训练集中至少N次的，使用"<unk>"替换出现次数少于N的
    Args:
        train_data, test_data: 列表，每个列表是一个句子的单词列表
        count_threshold: Words whose count is less than this are
                      treated as unknown.

    Returns:
        Tuple of
        - training data with low frequent words replaced by "<unk>"
        - test data with low frequent words replaced by "<unk>"
        - vocabulary of words that appear n times or more in the training data
    """

    vocabulary = get_words_with_nplus_frequency(train_data, count_threshold)

    # 使用"<unk>"替换训练集中出现次数低于N次的
    train_data_replaced = replace_oov_words_by_unk(train_data, vocabulary)

    # 使用"<unk>"替换训测试中出现次数低于N次的
    test_data_replaced = replace_oov_words_by_unk(test_data, vocabulary)

    return train_data_replaced, test_data_replaced, vocabulary


# 获取到预处理之后的测试集、训练集以及字典。vocabulary是一个词汇列表，出现次数大于N次的
minimum_freq = 2
train_data_processed, test_data_processed, vocabulary = preprocess_data(train_data,
                                                                        test_data, minimum_freq)

'''
    Part2:开发n-gram语言模型，得到所有单词的概率矩阵
'''
def count_n_grams(data, n, start_token='<s>', end_token='<e>'):
    """
    计算数据的n-gram模型

    Args:
        data: List of lists of words
        n: number of words in a sequence

    Returns:
        A dictionary that maps a tuple of n-words to its frequency
    """

    n_grams = {}

    for sentence in data:

        # 添加起始标记和结束标记
        sentence = [start_token] * n + sentence + [end_token]

        # 将每一个句子的单词列表，转为元祖，便于作key。形如 ('<s>', 'i', 'like', 'a', 'cat', '<e>')
        sentence = tuple(sentence)

        for i in range(len(sentence)):

            n_gram = sentence[i:i+n]

            if n_gram in n_grams:
                n_grams[n_gram] += 1
            else:
                n_grams[n_gram] = 1

    return n_grams


def estimate_probability(word, previous_n_gram,
                         n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
    """
    使用n-gram计数和k平滑评估下一个单词出现的概率

    Args:
        word: 下一个单词，比如"cat"
        previous_n_gram: 长度为n的单词序列， 比如["a"]
        n_gram_counts: n-grams的计数字典
        n_plus1_gram_counts:(n+1)-grams的计数字典，在n_gram_counts的基础上增加一个字符维度
        vocabulary_size: 词汇表大小，即单词数量
        k: 正数常量，平滑值

    Returns:
        A probability
    """
    # 列表转为词组
    previous_n_gram = tuple(previous_n_gram)

    # 分母
    previous_n_gram_count = n_gram_counts[previous_n_gram] if previous_n_gram in n_gram_counts else 0
    denominator = previous_n_gram_count + k * vocabulary_size

    # 结合previous_n_gram_count和word，定义n+1序列

    n_plus1_gram = previous_n_gram + (word, )

    # 获取字典中当前单词序列的计数值
    n_plus1_gram_count = n_plus1_gram_counts[n_plus1_gram] if n_plus1_gram in n_plus1_gram_counts else 0

    # 分子
    numerator = n_plus1_gram_count + k

    probability = numerator / denominator

    return probability


def estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, k=1.0):
    """
    评估所有单词的概率
    Args:
        previous_n_gram: 长度为n的单词序列， 比如["a"]
        n_gram_counts: n-grams的计数字典
        n_plus1_gram_counts: (n+1)-grams的计数字典，在n_gram_counts的基础上增加一个字符维度
        vocabulary: 词汇表
        k: 正数常量，平滑值
    Returns:
        A dictionary mapping from next words to the probability.
    """

    # list转tuple
    previous_n_gram = tuple(previous_n_gram)

    # 在字典中添加两个标记， <e> 和 <unk>，<s>不会出现在下一个单词中，因此不需要
    vocabulary = vocabulary + ["<e>", "<unk>"]
    vocabulary_size = len(vocabulary)

    probabilities = {}
    for word in vocabulary:
        probability = estimate_probability(word, previous_n_gram,
                                           n_gram_counts, n_plus1_gram_counts,
                                           vocabulary_size, k=k)
        probabilities[word] = probability

    return probabilities


def make_count_matrix(n_plus1_gram_counts, vocabulary):
    '''
    构建计数矩阵
    :param n_plus1_gram_counts:
    :param vocabulary:
    :return:
    '''

    vocabulary = vocabulary + ["<e>", "<unk>"]

    # obtain unique n-grams
    n_grams = []
    for n_plus1_gram in n_plus1_gram_counts.keys():
        n_gram = n_plus1_gram[0:-1]
        n_grams.append(n_gram)
    n_grams = list(set(n_grams))

    # mapping from n-gram to row
    row_index = {n_gram: i for i, n_gram in enumerate(n_grams)}
    # mapping from next word to column
    col_index = {word: j for j, word in enumerate(vocabulary)}

    nrow = len(n_grams)
    ncol = len(vocabulary)
    count_matrix = np.zeros((nrow, ncol))
    for n_plus1_gram, count in n_plus1_gram_counts.items():
        n_gram = n_plus1_gram[0:-1]
        word = n_plus1_gram[-1]
        if word not in vocabulary:
            continue
        i = row_index[n_gram]
        j = col_index[word]
        count_matrix[i, j] = count

    count_matrix = pd.DataFrame(count_matrix, index=n_grams, columns=vocabulary)
    return count_matrix

def make_probability_matrix(n_plus1_gram_counts, vocabulary, k):
    '''
    构建概率矩阵：将计数矩阵中各元素除以各行总数和得到概率矩阵
    :param n_plus1_gram_counts:
    :param vocabulary:
    :param k:
    :return:
    '''
    count_matrix = make_count_matrix(n_plus1_gram_counts, vocabulary)
    count_matrix += k
    prob_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)
    return prob_matrix


'''
    Part3：计算困惑度
'''
def calculate_perplexity(sentence, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
    """
    计算句子列表的困惑度

    Args:
        sentence: List of strings
        n_gram_counts: Dictionary of counts of n-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary_size: number of unique words in the vocabulary
        k: Positive smoothing constant

    Returns:
        Perplexity score
    """
    # 先前的单词长度
    n = len(list(n_gram_counts.keys())[0])

    # 添加起始标记和终止标记
    sentence = ["<s>"] * n + sentence + ["<e>"]

    # 转换成元组
    sentence = tuple(sentence)

    # 添加标记后的句子长度
    N = len(sentence)

    # 初始化值，用以后续的困惑度累乘
    product_pi = 1.0

    for t in range(n, N):

        # n表示先前的单词长度，每次从句子中取这个长度的字符
        n_gram = sentence[t-n:t]
        # 位置t的单词
        word = sentence[t]

        # 评估下一个单词出现的可能性
        probability = estimate_probability(word, n_gram,
                         n_gram_counts, n_plus1_gram_counts, vocabulary_size, k)


        product_pi *= 1 / probability

    # 困惑度计算
    perplexity = product_pi**(1/float(N))

    return perplexity


'''
    Part4：创建自动补全系统
'''


def suggest_a_word(previous_tokens, n_gram_counts, n_plus1_gram_counts, vocabulary, k=1.0, start_with=None):
    """
    为下一个出现的单词给出建议

    Args:
        previous_tokens: The sentence you input where each token is a word. Must have length > n
        n_gram_counts: Dictionary of counts of (n+1)-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary: List of words
        k: positive constant, smoothing parameter
        start_with: If not None, specifies the first few letters of the next word

    Returns:
        A tuple of
          - string of the most likely next word
          - corresponding probability
    """

    # 先前单词的长度
    n = len(list(n_gram_counts.keys())[0])

    # 从用户输入的单词中获取最近的 “n” 个单词作为前一个 n-gram。
    previous_n_gram = previous_tokens[-n:]

    # 估计词汇表中每个单词是下一个单词的概率。
    probabilities = estimate_probabilities(previous_n_gram,
                                           n_gram_counts, n_plus1_gram_counts,
                                           vocabulary, k=k)

    # 初始化
    suggestion = None
    max_prob = 0

    for word, prob in probabilities.items():

        if start_with != None:

            if not word.startswith(start_with):
                continue

        if prob > max_prob:

            suggestion = word

            max_prob = prob

    return suggestion, max_prob


def get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0, start_with=None):
    model_counts = len(n_gram_counts_list)
    suggestions = []
    for i in range(model_counts - 1):
        n_gram_counts = n_gram_counts_list[i]
        n_plus1_gram_counts = n_gram_counts_list[i + 1]

        suggestion = suggest_a_word(previous_tokens, n_gram_counts,
                                    n_plus1_gram_counts, vocabulary,
                                    k=k, start_with=start_with)
        suggestions.append(suggestion)
    return suggestions


'''
    Part5：测试
'''
n_gram_counts_list = []
for n in range(1, 6):
    print("Computing n-gram counts with n =", n, "...")
    n_model_counts = count_n_grams(train_data_processed, n)
    n_gram_counts_list.append(n_model_counts)

n_gram_counts_list = []
for n in range(1, 6):
    print("Computing n-gram counts with n =", n, "...")
    n_model_counts = count_n_grams(train_data_processed, n)
    n_gram_counts_list.append(n_model_counts)
#%%
previous_tokens = ["i", "am", "to"]
tmp_suggest4 = get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0)

print(f"The previous words are {previous_tokens}, the suggestions are:")
print(tmp_suggest4)

previous_tokens = ["i", "want", "to", "go"]
tmp_suggest5 = get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0)

print(f"The previous words are {previous_tokens}, the suggestions are:")
print(tmp_suggest5)
#%%
previous_tokens = ["hey", "how", "are"]
tmp_suggest6 = get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0)

print(f"The previous words are {previous_tokens}, the suggestions are:")
print(tmp_suggest6)
#%%
previous_tokens = ["hey", "how", "are", "you"]
tmp_suggest7 = get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0)

print(f"The previous words are {previous_tokens}, the suggestions are:")
print(tmp_suggest7)
#%%
previous_tokens = ["hey", "how", "are", "you"]
tmp_suggest8 = get_suggestions(previous_tokens, n_gram_counts_list, vocabulary, k=1.0, start_with="d")

print(f"The previous words are {previous_tokens}, the suggestions are:")
print(tmp_suggest8)