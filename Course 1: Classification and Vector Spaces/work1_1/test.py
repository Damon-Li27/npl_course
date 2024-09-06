import numpy as np

from nltk.corpus import twitter_samples
import nltk
from utils import process_tweet, build_freqs

from utils import process_tweet, build_freqs

'''
    nltk是自然语言工具包，语料库下载失败的话，可以手动下载：https://www.nltk.org/nltk_data/
'''
nltk.download('twitter_samples')

nltk.download('stopwords')

# 从下载的twitter_samples数据集中读取两个 JSON 文件中的内容
all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# 将正负面的推文分别划分为训练集和测试集。前 4000 条正面推文作为训练集，剩余的作为测试集；同样，前 4000 条负面推文作为训练集，剩余的作为测试集。
test_pos = all_positive_tweets[4000:] # 从索引4000开始一直到最后
train_pos = all_positive_tweets[:4000] # 从头一直取到3999的位置（不包含索引为4000的元素）
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

#对于训练集，前面是长度为len(train_pos)的全为 1 的数组（代表正面推文的标签），
# 后面接着长度为len(train_neg)的全为 0 的数组（代表负面推文的标签），并沿着行的方向（axis=0）拼接起来。
train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

print("train_y.shape = " + str(train_y.shape)) # (8000, 1)
print("test_y.shape = " + str(test_y.shape)) # (2000, 1)

# 构建一个频率字典，键为单词和对应的标签组成的元组，值是该单词在特定标签下出现的次数。
freqs = build_freqs(train_x, train_y)

# check the output
print("type(freqs) = " + str(type(freqs)))
print("len(freqs) = " + str(len(freqs.keys()))) # 11337



def sigmoid(z):
    h = 1 / (1 + np.exp(-z))
    return h

def cost(y, y_hat):
    return -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def gradientDescent(x, y, theta, alpha, num_iters):

    m = len(x)

    for i in range(0, num_iters):
        # get z, the dot product of x and theta
        # x:(8000, 3), theta:(3,1)
        z = np.dot(x, theta)

        # get the sigmoid of z
        h = sigmoid(z)

        # calculate the cost function
        J = 1 / m * np.sum(cost(y, h))

        if (i % 100 == 0):
            print(f'after {i} iterations, the loss value is : {J}')
        # update the weights theta
        d_theta = 1 / m * np.dot(x.T, h - y)
        theta = theta - alpha * d_theta

    ### END CODE HERE ###
    J = float(J)
    return J, theta


def extract_features(tweet, freqs):
    word_l = process_tweet(tweet)

    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros((1, 3))

    # bias term is set to 1
    x[0, 0] = 1

    for word in word_l:
        # increment the word count for the positive label 1
        x[0, 1] += freqs.get((word, 1.0), 0)

        # increment the word count for the negative label 0
        x[0, 2] += freqs.get((word, 0), 0)

    ### END CODE HERE ###
    assert (x.shape == (1, 3))
    return x


def train_model():
    # 根据训练集的长度初始化矩阵
    X = np.zeros((len(train_x), 3))
    for i in range(len(train_x)):
        # 从训练集中提取特征，将其转化为矩阵
        X[i, :] = extract_features(train_x[i], freqs)
    Y = train_y

    J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 5000)
    print(f"The cost after training is {J:.8f}.")
    print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")
    return theta


def predict_tweet(tweet, freqs, theta):
    '''
        Input:
            tweet: a string
            freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
            theta: (3,1) vector of weights
        Output:
            y_pred: the probability of a tweet being positive or negative
        '''

    x = extract_features(tweet, freqs)

    # make the prediction using x and theta
    y_pred = sigmoid(np.dot(x, theta))

    ### END CODE HERE ###

    return y_pred


def test_logistic_regression(test_x, test_y, freqs, theta):
    """
    Input:
        test_x: a list of tweets
        test_y: (m, 1) vector with the corresponding labels for the list of tweets
        freqs: a dictionary with the frequency of each pair (or tuple)
        theta: weight vector of dimension (3, 1)
    Output:
        accuracy: (# of tweets classified correctly) / (total # of tweets)
    """

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

    # the list for storing predictions
    y_hat = []

    for tweet in test_x:
        # get the label prediction for the tweet
        y_pred = predict_tweet(tweet, freqs, theta)

        if y_pred > 0.5:
            # append 1.0 to the list
            y_hat.append(1.0)
        else:
            # append 0 to the list
            y_hat.append(0.0)

    # With the above implementation, y_hat is a list, but test_y is (m,1) array
    # convert both to one-dimensional arrays in order to compare them using the '==' operator
    accuracy = 1 / len(test_x) * np.sum(y_hat == np.squeeze(test_y))

    ### END CODE HERE ###

    return accuracy

def test(theta, freqs):
    for tweet in ['I am happy', 'I am bad', 'this movie should have been great.', 'this movie should have been bad.',
                  'great', 'great great', 'great great great', 'great great great great', 'I am learning :)']:
        # '%s -> %f'是一个格式化字符串，其中 %s 表示要插入一个字符串，%f 表示要插入一个浮点数。
        print('%s -> %f' % (tweet, predict_tweet(tweet, freqs, theta)))

    my_tweet = 'This is a ridiculously bright movie. The plot was terrible and I was sad until the ending!'
    print(process_tweet(my_tweet))
    y_hat = predict_tweet(my_tweet, freqs, theta)
    print(y_hat)
    if y_hat > 0.5:
        print('Positive sentiment')
    else:
        print('Negative sentiment')


# 训练
theta = train_model()

# 测试集
accuracy = test_logistic_regression(test_x, test_y, freqs, theta)
print("accuracy:", accuracy)

# 测试
test(theta, freqs)



