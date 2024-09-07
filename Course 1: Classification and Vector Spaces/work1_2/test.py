from nltk.corpus import stopwords, twitter_samples
import numpy as np
import nltk
from utils import process_tweet, lookup

nltk.download('stopwords')
nltk.download('twitter_samples')

# 从下载的twitter_samples数据集中读取两个 JSON 文件中的内容
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# 将正负面的推文分别划分为训练集和测试集。前 4000 条正面推文作为训练集，剩余的作为测试集；同样，前 4000 条负面推文作为训练集，剩余的作为测试集。
test_pos = all_positive_tweets[4000:]  # 从索引4000开始一直到最后
train_pos = all_positive_tweets[:4000]  # 从头一直取到3999的位置（不包含索引为4000的元素）
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

# 对于训练集，前面是长度为len(train_pos)的全为 1 的数组（代表正面推文的标签），
# 后面接着长度为len(train_neg)的全为 0 的数组（代表负面推文的标签），并沿着行的方向（axis=0）拼接起来。
train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

print("train_y.shape = " + str(train_y.shape))  # (8000, 1)
print("test_y.shape = " + str(test_y.shape))  # (2000, 1)


def count_tweets(result, tweets, ys):
    '''
    生成词频字典 - result
    Input:
        result: a dictionary that will be used to map each pair to its frequency
        tweets: a list of tweets
        ys: a list corresponding to the sentiment of each tweet (either 0 or 1)
    Output:
        result: a dictionary mapping each pair to its frequency
    '''

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    yslist = np.squeeze(ys).tolist()
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            # define the key, which is the word and label tuple
            pair = (word, y)

            # if the key exists in the dictionary, increment the count
            if pair in result:
                result[pair] += 1

            # else, if the key is new, add it to the dictionary and set the count to 1
            else:
                result[pair] = 1
    ### END CODE HERE ###

    return result


# 创建字典
freqs = count_tweets({}, train_x, train_y)


def train_naive_bayes(freqs, train_x, train_y):
    '''
    训练模型
    Input:
        freqs: dictionary from (word, label) to how often the word appears
        train_x: a list of tweets
        train_y: a list of labels correponding to the tweets (0,1)
    Output:
        logprior: the log prior. (equation 3 above)
        loglikelihood: the log likelihood of you Naive bayes equation. (equation 6 above)
    '''
    loglikelihood = {}
    logprior = 0


    # vocab为所有word的词典集合,
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)

    # 计算推文总数
    D = len(train_x)

    # 计算正向推文总数
    D_pos = np.sum(train_y == 1)

    # 计算负向推文总数
    D_neg = np.sum(train_y == 0)

    # Calculate logprior 先验概率
    logprior = np.log(D_pos) - np.log(D_neg)

    # For each word in the vocabulary...
    for word in vocab:
        # get the positive and negative frequency of the word - 每个词的正向和负向频率
        freq_pos = lookup(freqs, word, 1)
        freq_neg = lookup(freqs, word, 0)

        # calculate the probability that each word is positive, and negative
        p_w_pos = (freq_pos + 1) / (D_pos + V)
        p_w_neg = (freq_neg + 1) / (D_neg + V)

        # calculate the log likelihood of the word - 即lambda
        loglikelihood[word] = np.log(p_w_pos / p_w_neg)

    return logprior, loglikelihood


logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)


# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def naive_bayes_predict(tweet, logprior, loglikelihood):
    '''
    计算对数似然比，即lambda
    Input:
        tweet: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Output:
        p: the sum of all the logliklihoods of each word in the tweet (if found in the dictionary) + logprior (a number)

    '''
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    # process the tweet to get a list of words
    word_l = process_tweet(tweet)

    # initialize probability to zero
    p = 0

    # add the logprior
    p += logprior

    for word in word_l:

        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word to the probability
            p += loglikelihood[word]

    ### END CODE HERE ###

    return p


# UNQ_C6 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def test_naive_bayes(test_x, test_y, logprior, loglikelihood):
    """
    检测预测的准确性
    Input:
        test_x: A list of tweets
        test_y: the corresponding labels for the list of tweets
        logprior: the logprior
        loglikelihood: a dictionary with the loglikelihoods for each word
    Output:
        accuracy: (# of tweets classified correctly)/(total # of tweets)
    """
    accuracy = 0  # return this properly

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    y_hats = []
    for tweet in test_x:
        # if the prediction is > 0
        if naive_bayes_predict(tweet, logprior, loglikelihood) > 0:
            # the predicted class is 1
            y_hat_i = 1
        else:
            # otherwise the predicted class is 0
            y_hat_i = 0

        # append the predicted class to the list y_hats
        y_hats.append(y_hat_i)

    # error is the average of the absolute values of the differences between y_hats and test_y
    error = np.sum(np.abs(np.squeeze(test_y) - y_hats)) / len(np.squeeze(test_y))

    # Accuracy is 1 minus the error
    accuracy = 1 - error

    ### END CODE HERE ###

    return accuracy

accuracy = test_naive_bayes(test_x, test_y, logprior, loglikelihood)
print("Accuracy of test database:", accuracy)


def test():
    my_tweet = 'She smiled.'
    p = naive_bayes_predict(my_tweet, logprior, loglikelihood)
    print('The expected output is', p)
    print(f"Naive Bayes accuracy = {test_naive_bayes(test_x, test_y, logprior, loglikelihood)}")

    for tweet in ['I am happy', 'I am bad', 'this movie should have been great.', 'great', 'great great',
                  'great great great', 'great great great great']:
        # print( '%s -> %f' % (tweet, naive_bayes_predict(tweet, logprior, loglikelihood)))
        p = naive_bayes_predict(tweet, logprior, loglikelihood)
        #     print(f'{tweet} -> {p:.2f} ({p_category})')
        print(f'{tweet} -> {p:.2f}')

    my_tweet = 'you are bad :('
    naive_bayes_predict(my_tweet, logprior, loglikelihood)


def get_ratio(freqs, word):
    '''
    给出 freqs 字典和任意个单词，使用lookup(freqs,word,1) 查看该单词的正负向计数以及笔记（判断该单词在哪个语料库中出现的次数多）

    Input:
        freqs: dictionary containing the words
        word: string to lookup

    Output: a dictionary with keys 'positive', 'negative', and 'ratio'.
        Example: {'positive': 10, 'negative': 20, 'ratio': 0.5}
    '''
    pos_neg_ratio = {'positive': 0, 'negative': 0, 'ratio': 0.0}
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    # use lookup() to find positive counts for the word (denoted by the integer 1)
    pos_neg_ratio['positive'] = lookup(freqs, word, 1)

    # use lookup() to find negative counts for the word (denoted by integer 0)
    pos_neg_ratio['negative'] = lookup(freqs, word, 0)

    # calculate the ratio of positive to negative counts for the word
    pos_neg_ratio['ratio'] = (pos_neg_ratio['positive'] + 1) / (pos_neg_ratio['negative'] + 1)
    ### END CODE HERE ###
    return pos_neg_ratio


# UNQ_C9 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def get_words_by_threshold(freqs, label, threshold):
    '''
    构建一个字典(以threshold为门槛进行统计)，其中键为word，值为一个字典类型pos_neg_ratio，即get_ratio()的返回值
    当 label 设为1， 选择正负计数比大于等于阈值的单词
    当 label 设为0， 选择正负计数比小于等于阈值的单词
    Input:
        freqs: dictionary of words
        label: 1 for positive, 0 for negative
        threshold: ratio that will be used as the cutoff for including a word in the returned dictionary
    Output:
        word_set: dictionary containing the word and information on its positive count, negative count, and ratio of positive to negative counts.
        example of a key value pair:
        {'happi':
            {'positive': 10, 'negative': 20, 'ratio': 0.5}
        }
    '''
    word_list = {}

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    for key in freqs.keys():
        word, _ = key

        # get the positive/negative ratio for a word
        pos_neg_ratio = get_ratio(freqs, word)

        # if the label is 1 and the ratio is greater than or equal to the threshold...
        if label == 1 and pos_neg_ratio['ratio'] > threshold:

            # Add the pos_neg_ratio to the dictionary
            word_list[word] = pos_neg_ratio

        # If the label is 0 and the pos_neg_ratio is less than or equal to the threshold...
        elif label == 0 and pos_neg_ratio['ratio'] < threshold:

            # Add the pos_neg_ratio to the dictionary
            word_list[word] = pos_neg_ratio

        # otherwise, do not include this word in the list (do nothing)

    ### END CODE HERE ###
    return word_list


def find_error():
    # 找出那些模型预测错误的推特。为什么会出错？朴素贝叶斯模型有什么假设吗？
    print('Truth Predicted Tweet')
    for x, y in zip(test_x, test_y):
        y_hat = naive_bayes_predict(x, logprior, loglikelihood)
        if y != (np.sign(y_hat) > 0):
            print('%d\t%0.2f\t%s' % (y, np.sign(y_hat) > 0, ' '.join(
                process_tweet(x)).encode('ascii', 'ignore')))


my_tweet = 'I am happy because I am learning :)'

p = naive_bayes_predict(my_tweet, logprior, loglikelihood)
print(p)
