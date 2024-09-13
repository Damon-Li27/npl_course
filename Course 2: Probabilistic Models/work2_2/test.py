from utils_pos import get_word_tag, preprocess
import pandas as pd
from collections import defaultdict
import math
import numpy as np
from utils_pos import preprocess

with open('WSJ_02-21.pos', 'r') as f:
    # 训练集字典，key为word，value为词性标签
    # training_corpus：训练集语料库是由句子组成，每个句子的单词都带上了标签并做了换行处理。空一行，表示下一句话。
    training_corpus = f.readlines()

### START CODE HERE (处理测试集数据) ###
with open("WSJ_24.pos", 'r') as f:
    # 测试集字典，格式与训练集相同
    y = f.readlines()

with open("hmm_vocab.txt", 'r') as f:
    # 词汇表 - 列表
    voc_l = f.read().split('\n')

vocab = {}

# 创建对应单词的索引字典
for i, word in enumerate(sorted(voc_l)):
    vocab[word] = i

# 预处理，test.words是测试集去除标签后生成的单词列表
_, prep = preprocess(vocab, "test.words")


### END CODE HERE ###

def create_dictionaries(training_corpus, vocab):
    """
    创建字典：状态转移概率字典、发射概率字典、标签数量字典
    Input:
        training_corpus: a corpus where each line has a word followed by its tag.
        vocab: a dictionary where keys are words in vocabulary and value is an index
    Output:
        emission_counts: a dictionary where the keys are (tag, word) and the values are the counts
        transition_counts: a dictionary where the keys are (prev_tag, tag) and the values are the counts
        tag_counts: a dictionary where the keys are the tags and the values are the counts
    """

    # 初始化矩阵，defaultdict(int) 会在访问不存在的键时返回0
    emission_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    tag_counts = defaultdict(int)

    # 起始状态（以 “--s--” 表示）,连接每句话的第一个单词。
    prev_tag = '--s--'

    # i 追踪语料库的行号(即word: tag数量)
    i = 0

    # 遍历训练集，每条数据包含一个word和它对应的词性标签
    for word_tag in training_corpus:

        # word: tag数量
        i += 1

        # 每 50,000 单词打印一次单词数量
        if i % 50000 == 0:
            print(f"word count = {i}")

        word, tag = get_word_tag(word_tag, vocab)

        transition_counts[(prev_tag, tag)] += 1

        emission_counts[(tag, word)] += 1

        tag_counts[tag] += 1

        prev_tag = tag

    return emission_counts, transition_counts, tag_counts


emission_counts, transition_counts, tag_counts = create_dictionaries(training_corpus, vocab)

states = sorted(tag_counts.keys())


def predict_pos(prep, y, emission_counts, vocab, states):
    '''
    prep是test.word中所有进行过预处理的单词列表。遍历该单词列表，
    在上面构建的发射矩阵中找到该单词对应的最大可能的词性标签，看它是否与测试集该单词对应的标签相同。
    以此，评估这种方法的效果如何。
    Input:
        prep: “y” 的预处理版本。一个组成元组的 “word”（单词）的列表。（测试集去除标签后生成的单词列表）
        y: 一个语料库，由一个元组列表组成，其中每个元组由（word，POS）（单词，词性）组成。
        emission_counts: a dictionary where the keys are (tag,word) tuples and the value is the count
        vocab: a dictionary where keys are words in vocabulary and value is an index
        states: 一个已排序的列表，包含所有可能的词性。
    Output:
        accuracy: Number of times you classified a word correctly
    '''

    num_correct = 0

    # （单词，词性）数
    total = len(y)
    # zip用于将多个可迭代对象（如列表、元组、字符串等）组合成一个元组序列。如果多个可迭代对象的长度不一致，zip会以最短的可迭代对象为准进行组合
    for word, y_tup in zip(prep, y):

        # split函数是字符串对象的一个方法，用于将字符串分割成一个列表。将 (word, POS) 分割为两个字符串列表
        y_tup_l = y_tup.split()

        # 判断 y_tup 是否包含word和pos
        if len(y_tup_l) == 2:
            true_label = y_tup_l[1]
        else:
            continue

        count_final = 0
        pos_final = ''

        if word in vocab:
            for pos in states:

                key = (pos, word)

                # 查找该单词对应的可能性最大的词性标签
                if key in emission_counts:
                    count = emission_counts.get(key)

                    if count > count_final:
                        count_final = count

                        pos_final = pos

            # 如果查找到的词性标签，与测试集的词性标签相同，即正确
            if pos_final == true_label:
                num_correct += 1

    accuracy = num_correct / total

    return accuracy


accuracy_predict_pos = predict_pos(prep, y, emission_counts, vocab, states)

'''
 创建状态转移矩阵和发射矩阵
'''
def create_transition_matrix(alpha, tag_counts, transition_counts):
    '''
    利用transition_counts，构建状态转移矩阵
    Input:
        alpha: 平滑的值
        tag_counts: 训练集中，所有标签及其对应数量的字典。key为tag，值为count。
        transition_counts: key为（prev_tag, tag），值为count
    Output:
        A: matrix of dimension (num_tags,num_tags)
    '''

    all_tags = sorted(tag_counts.keys())
    num_tags = len(all_tags)  # 标签个数
    # 初始化矩阵
    A = np.zeros((num_tags, num_tags))

    for i in range(num_tags):

        for j in range(num_tags):

            count = 0

            # 定义key
            key = (all_tags[i], all_tags[j])
            if key in transition_counts:
                count = transition_counts[key]

            count_prev_tag = tag_counts.get(all_tags[i])

            A[i, j] = (count + alpha) / (count_prev_tag + alpha * num_tags)
    return A

def create_emission_matrix(alpha, tag_counts, emission_counts, all_words):
    '''
    创建发射概率矩阵
    Input:
        alpha: 平滑的值
        tag_counts: 训练集中，所有标签及其对应数量的字典。key为tag，值为count。
        emission_counts: key为(tag, word)，value为数量的字典
        all_words: 单词列表
    Output:
        B: a matrix of dimension (num_tags, len(vocab))
    '''

    # 词性标签个数
    num_tags = len(tag_counts)
    # 所有词性标签列表
    all_tags = sorted(tag_counts.keys())

    # 单词个数
    num_words = len(vocab)

    # 初始化发射矩阵，行为标签数，列为单词个数
    B = np.zeros((num_tags, num_words))

    for i in range(num_tags):
        for j in range(num_words):
            count = 0
            key = (all_tags[i], all_words[j])
            if key in emission_counts:
                count = emission_counts.get(key)

            count_tag = tag_counts.get(all_tags[i])

            B[i, j] = (count + alpha) / (count_tag + num_words * alpha)
    return B

alpha = 0.001
A = create_transition_matrix(alpha, tag_counts, transition_counts)
B = create_emission_matrix(alpha, tag_counts, emission_counts, list(vocab))

'''
    维比特算法：
    初始化：首先初始化两个辅助矩阵best_probs和best_paths，根据初始标记字符'--s--'，判断哪些标签可以作为文本的首单词（与'--s--'相连的），
    在训练集中，没有出现过（即概率为0）的，记为负无穷，否则计算出初始状态到第一个发射单词的概率。
    
'''


def initialize(states, tag_counts, A, B, corpus, vocab):
    '''
    Input:
        states: tag_counts的keys()，即所有词性标签
        tag_counts: 训练集中，所有标签及其对应数量的字典。key为tag，值为count。
        A: 状态转移矩阵A，维度为(标签个数，标签个数)
        B: 发射矩阵B，维度为 (标签个数, 单词个数)
        corpus: 一个组成元组的 “word”（单词）的列表。（测试集去除标签后生成的单词列表）
        vocab:创建对应单词的索引字典
    Output:
        best_probs: matrix of dimension (num_tags, len(corpus)) of floats
        best_paths: matrix of dimension (num_tags, len(corpus)) of integers
    '''

    num_tags = len(tag_counts)

    # 初始化两个辅助矩阵，行数为标签数，列数为单词数
    best_probs = np.zeros((num_tags, len(corpus)))  # 用于存储每个状态在每个位置的最佳概率。
    best_paths = np.zeros((num_tags, len(corpus)), dtype=int)  # 用于存储每个状态在每个位置的最佳路径。

    #  “--s--” 通常是一个特殊的标记，用于表示序列（在这个例子中是词性标注序列）的起始状态。(一个不太可能在正常文本中出现的特殊字符串)
    # 在函数create_dictionaries中，使用“--s--”连接首字母
    s_idx = states.index("--s--")

    for i in range(num_tags):
        # 如果s_idx对应的某列中，概率为 0，表示在这个一段文本在这个隐藏状态（词性标签）下开始的概率极低，将best_probs[i, 0]设置为负无穷大。
        if A[s_idx, i] == 0:
            best_probs[i, 0] = float('-inf')

        # 如果概率不为 0，则计算起始标识到该状态的转移概率与该状态下生成第一个单词的发射概率之和，并将其存储在best_probs[i, 0]中
        else:
            best_probs[i, 0] = math.log(A[s_idx, i]) + math.log(B[i, vocab[corpus[0]]])
    return best_probs, best_paths


best_probs, best_paths = initialize(states, tag_counts, A, B, prep, vocab)


def viterbi_forward(A, B, test_corpus, best_probs, best_paths, vocab):
    '''

    维特比算法前向计算

    Input:
        A, B: 转移矩阵与发射矩阵
        test_corpus: 预处理后的数据列表（测试集去除标签后生成的单词列表）
        best_probs: 最大概率矩阵，维度： (num_tags, len(corpus))
        best_paths: 最有路径矩阵，维度 (num_tags, len(corpus))
        vocab: 对应单词的索引字典
    Output:
        best_probs: a completed matrix of dimension (num_tags, len(corpus))
        best_paths: a completed matrix of dimension (num_tags, len(corpus))
    '''
    # 标签数量 (best_probs矩阵的行数)
    num_tags = best_probs.shape[0]

    # 遍历所有单词
    for i in range(1, len(test_corpus)):

        if i % 5000 == 0:
            print("Words processed: {:>8}".format(i))

        # 从当前单词的词性标签可到达的词性标签
        for j in range(num_tags):

            # 初始化单词i的最大概率为负无穷
            best_prob_i = float('-inf')

            # 初始化最佳路径
            best_path_i = None

            for k in range(num_tags):
                # 词性标签k的最大概率
                prob = best_probs[k,i-1] + math.log(A[k,j]) + math.log(B[j,vocab[test_corpus[i]]])
                if prob > best_prob_i:

                    best_prob_i = prob

                    best_path_i = k

            best_probs[j, i] = best_prob_i

            best_paths[j, i] = best_path_i

    return best_probs, best_paths

best_probs, best_paths = viterbi_forward(A, B, prep, best_probs, best_paths, vocab)


def viterbi_backward(best_probs, best_paths, corpus, states):
    '''
    This function returns the best path.

    '''
    # Get the number of words in the corpus
    # which is also the number of columns in best_probs, best_paths
    m = best_paths.shape[1]

    # Initialize array z, same length as the corpus
    z = [None] * m

    # Get the number of unique POS tags
    num_tags = best_probs.shape[0]

    # Initialize the best probability for the last word
    best_prob_for_last_word = float('-inf')

    # Initialize pred array, same length as corpus
    pred = [None] * m

    ### START CODE HERE (Replace instances of 'None' with your code) ###
    ## Step 1 ##

    # Go through each POS tag for the last word (last column of best_probs)
    # in order to find the row (POS tag integer ID)
    # with highest probability for the last word
    for k in range(num_tags):  # complete this line

        # If the probability of POS tag at row k
        # is better than the previously best probability for the last word:
        if best_probs[k, m - 1] > best_prob_for_last_word:  # complete this line

            # Store the new best probability for the last word
            best_prob_for_last_word = best_probs[k, m - 1]

            # Store the unique integer ID of the POS tag
            # which is also the row number in best_probs
            z[m - 1] = k

    # Convert the last word's predicted POS tag
    # from its unique integer ID into the string representation
    # using the 'states' list
    # store this in the 'pred' array for the last word
    pred[m - 1] = states[z[m - 1]]

    ## Step 2 ##
    # Find the best POS tags by walking backward through the best_paths
    # From the last word in the corpus to the 0th word in the corpus
    for i in range(m - 1, 0, -1):  # complete this line

        # Retrieve the unique integer ID of
        # the POS tag for the word at position 'i' in the corpus
        pos_tag_for_word_i = best_paths[z[i], i]

        # In best_paths, go to the row representing the POS tag of word i
        # and the column representing the word's position in the corpus
        # to retrieve the predicted POS for the word at position i-1 in the corpus
        z[i - 1] = pos_tag_for_word_i

        # Get the previous word's POS tag in string form
        # Use the 'states' list,
        # where the key is the unique integer ID of the POS tag,
        # and the value is the string representation of that POS tag
        pred[i - 1] = states[z[i - 1]]

    ### END CODE HERE ###
    return pred

pred = viterbi_backward(best_probs, best_paths, prep, states)

def compute_accuracy(pred, y):
    '''
    Input:
        pred: a list of the predicted parts-of-speech
        y: a list of lines where each word is separated by a '\t' (i.e. word \t tag)
    Output:

    '''
    num_correct = 0
    total = 0

    # Zip together the prediction and the labels
    for prediction, y in zip(pred, y):
        ### START CODE HERE (Replace instances of 'None' with your code) ###
        # Split the label into the word and the POS tag
        word_tag_tuple = y.split()

        # Check that there is actually a word and a tag
        # no more and no less than 2 items
        if len(word_tag_tuple) < 2:  # complete this line
            continue

            # store the word and tag separately
        word, tag = word_tag_tuple[0], word_tag_tuple[1]

        # Check if the POS tag label matches the prediction
        if tag == prediction:  # complete this line

            # count the number of times that the prediction
            # and label match
            num_correct += 1

        # keep track of the total number of examples (that have valid labels)
        total += 1

        ### END CODE HERE ###
    return num_correct / total


# %%
print(f"Accuracy of the Viterbi algorithm is {compute_accuracy(pred, y):.4f}")