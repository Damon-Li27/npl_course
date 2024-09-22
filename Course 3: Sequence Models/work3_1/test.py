import os
import random as rnd

import trax
import trax.fastmath as fastmath
import trax.fastmath.numpy as np
from trax import layers as tl
from utils import Layer, load_tweets, process_tweet
from trax.supervised import training
'''
    使用trax库训练，分析文本情感
'''


'''
    Part1：加载数据。 
    注意：每次执行代码需要现将model下内容删除。
'''
all_positive_tweets, all_negative_tweets = load_tweets()

# 分别取正向情绪和负向情绪的前4000条数据拼接作为训练数据，剩余数据拼接作为验证集数据，数据类型均为列表
val_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]

val_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
val_x = val_pos + val_neg

train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))

val_y = np.append(np.ones(len(val_pos)), np.zeros(len(val_neg)))

# 初始化Vocab
Vocab = {'__PAD__': 0, '__</e>__': 1, '__UNK__': 2}

# 为每一个单词编号，从0递增。得到的字典样式为：
# {'__PAD__': 0,
#  '__</e>__': 1,
#  '__UNK__': 2,
#  'followfriday': 3,
#  'top': 4,
#  'engag': 5, ...
for tweet in train_x:
    processed_tweet = process_tweet(tweet)
    for word in processed_tweet:
        if word not in Vocab:
            Vocab[word] = len(Vocab)


def tweet_to_tensor(tweet, vocab_dict, unk_token='__UNK__', verbose=False):
    '''
    将推文转成向量，不在字典中的字符用'__UNK__'的编码标记
    Input:
        tweet - A string containing a tweet
        vocab_dict - The words dictionary
        unk_token - The special string for unknown tokens
        verbose - Print info during runtime
    Output:
        tensor_l - 推文转换成的向量，比如推文为：
        Bro:U wan cut hair anot,ur hair long Liao bo
            Me:since ord liao,take it easy lor treat as save $ leave it longer :)
            Bro:LOL Sibei xialan
        将得到向量：[1064, 136, 478, 2351, 744, 8148, 1122, 744, 53, 2, 2671, 790, 2, 2, 348, 600, 2, 3488, 1016, 596, 4558, 9, 1064, 157, 2, 2]
    '''

    word_l = process_tweet(tweet)

    if verbose:
        print("List of words from the processed tweet:")
        print(word_l)

    tensor_l = []

    unk_ID = vocab_dict.get(unk_token)

    if verbose:
        print(f"The unique integer ID for the unk_token is {unk_ID}")

    for word in word_l:
        word_ID = vocab_dict.get(word, unk_ID)
        tensor_l.append(word_ID)

    return tensor_l

def data_generator(data_pos, data_neg, batch_size, loop, vocab_dict, shuffle=False):
    '''
    用于生成用于训练模型的批次数据，正负面各一半。
    Input:
        data_pos - 正面例子集合
        data_neg - 负面例子集合
        batch_size - 批量大小，必须为偶数
        loop - 一个布尔值，表示在遍历完数据集后是否循环继续使用数据。
        vocab_dict - 字典
        shuffle - 随机顺序
    Yield:
        inputs - Subset of positive and negative examples
        targets - The corresponding labels for the subset
        example_weights - An array specifying the importance of each example

    '''

    assert batch_size % 2 == 0

    # 在每一个批次中，正面例子和负面例子各占一半。因此，n_to_take表示正负样本的数量
    n_to_take = batch_size // 2

    # 初始化正、负样本的索引变量pos_index和neg_index，并获取正、负样本数据集的长度。
    # 创建正、负样本的索引列表，如果shuffle为True，则随机打乱这些索引列表。
    pos_index = 0
    neg_index = 0
    len_data_pos = len(data_pos)
    len_data_neg = len(data_neg)
    # 创建一个包含从 0 到len_data_pos - 1的整数列表。
    pos_index_lines = list(range(len_data_pos))
    neg_index_lines = list(range(len_data_neg))
    # 如果 “shuffle” 被设置为 “True”，则随机打乱行。
    if shuffle:
        # 使用随机数生成器 random分别对正样本索引列表 pos_index_lines 和负样本索引列表 neg_index_lines 进行随机打乱。
        # 这样做的目的可能是为了在后续处理数据时增加随机性，避免模型学习到特定的顺序模式，从而提高模型的泛化能力。
        rnd.shuffle(pos_index_lines)
        rnd.shuffle(neg_index_lines)

    # 无限循环生成批次数据，直到设置stop为True
    stop = False
    while not stop:
        batch = []  # 用于储存一个批次的样本
        # 添加一半的正样本
        for i in range(n_to_take):
            # 当正样本全部转为向量并添加到batch中后，判断loop，如果为False则跳出遍历正样本的循环，否则表示数据需要循环使用，重新随机化
            if pos_index >= len_data_pos:
                if not loop:
                    stop = True;
                    break;
                pos_index = 0
                if shuffle:
                    rnd.shuffle(pos_index_lines)
            tweet = data_pos[pos_index_lines[pos_index]]
            tensor = tweet_to_tensor(tweet, vocab_dict)
            batch.append(tensor)
            pos_index = pos_index + 1

        # 添加一半的负样本
        for i in range(n_to_take):
            if neg_index >= len_data_neg:
                if not loop:
                    stop = True;
                    break;
                neg_index = 0
                if shuffle:
                    rnd.shuffle(neg_index_lines)
            tweet = data_neg[neg_index_lines[neg_index]]
            tensor = tweet_to_tensor(tweet, vocab_dict)
            batch.append(tensor)
            neg_index = neg_index + 1
        if stop:
            break;

        pos_index += n_to_take

        neg_index += n_to_take

        # 处理批次数据长度和填充

        # 获取在这个批次中，最长的推文长度
        max_len = max([len(t) for t in batch])

        tensor_pad_l = []
        for tensor in batch:
            # 用0填充不足的数量，存入tensor_pad_l
            n_pad = max_len - len(tensor)
            pad_l = [0] * n_pad
            tensor_pad = tensor + pad_l
            tensor_pad_l.append(tensor_pad)
        # 转为numpy矩阵
        inputs = np.array(tensor_pad_l)

        # 定义标签列表，转为numpy矩阵
        target_pos = [1] * n_to_take
        target_neg = [0] * n_to_take
        target_l = target_pos + target_neg
        targets = np.array(target_l)

        # 设置示例权重为全 1，表示所有示例同等重要
        example_weights = np.ones_like(targets)

        yield inputs, targets, example_weights


def train_generator(batch_size, shuffle = False):
    '''
    生成批量训练数据
    :param batch_size:
    :param shuffle:
    :return:
    '''
    return data_generator(train_pos, train_neg, batch_size, True, Vocab, shuffle)

def val_generator(batch_size, shuffle = False):
    '''
    生成批量标签
    :param batch_size:
    :param shuffle:
    :return:
    '''
    return data_generator(val_pos, val_neg, batch_size, True, Vocab, shuffle)

def test_generator(batch_size, shuffle = False):
    '''
    生成批量测试数据
    :param batch_size:
    :param shuffle:
    :return:
    '''
    return data_generator(val_pos, val_neg, batch_size, False, Vocab, shuffle)

'''
    Part2: 定义class
'''
class Relu(Layer):
    """实现relu"""

    def forward(self, x):
        activation = np.maximum(0, x)
        return activation


class Dense(Layer):
    """
    实现密集层，继承自Layer
    """

    def __init__(self, n_units, init_stdev=0.1):
        '''
        :param n_units: 神经元数量
        :param init_stdev: 初始化权重的标准差
        '''
        self._n_units = n_units
        self._init_stdev = init_stdev

    def forward(self, x):
        dense = np.dot(x, self.weights)
        return dense

    # 初始化权重，重写了父类Layer的方法
    def init_weights_and_state(self, input_signature, random_key):
        '''
        :param input_signature: 输入的形状，第二个数值表示样本数
        :param random_key: 随机数生成器的键
        :return: 参数，其形状为(输入特征数量，神经元数量)
        '''
        input_shape = input_signature.shape
        w = self._init_stdev * fastmath.random.normal(key=random_key, shape=(input_shape[-1], self._n_units))
        self.weights = w
        return self.weights

def classifier(vocab_size=len(Vocab), embedding_dim=256, output_dim=2, mode='train'):
    # 创建嵌入层
    embed_layer = tl.Embedding(
        vocab_size=vocab_size,
        d_feature=embedding_dim)

    # 创建均值层，对序列进行均值化
    mean_layer = tl.Mean(axis=1)

    # 创建密集层
    dense_output_layer = tl.Dense(n_units=output_dim)

    # 创建对数softmax层
    log_softmax_layer = tl.LogSoftmax()

    model = tl.Serial(
        embed_layer,
        mean_layer,
        dense_output_layer,
        log_softmax_layer
    )
    return model


'''
    Part3：训练
'''
batch_size = 16
rnd.seed(271)

train_task = training.TrainTask(
    labeled_data=train_generator(batch_size=batch_size, shuffle=True),
    loss_layer=tl.CrossEntropyLoss(),
    optimizer=trax.optimizers.Adam(0.01),
    n_steps_per_checkpoint=10
)

eval_task = training.EvalTask(
    labeled_data=val_generator(batch_size=batch_size, shuffle=True),
    metrics=[tl.CrossEntropyLoss(), tl.Accuracy()]
)

def train_model(classifier, train_task, eval_task, n_steps, output_dir):
    '''
    模型训练
    Input:
        classifier - the model you are building
        train_task - Training task
        eval_task - Evaluation task
        n_steps - the evaluation steps
        output_dir - folder to save your files
    Output:
        trainer -  trax trainer
    '''
    training_loop = training.Loop(
                                classifier, # The learning model
                                train_task, # The training task
                                eval_tasks = [eval_task], # The evaluation task
                                output_dir = output_dir) # The output directory

    training_loop.run(n_steps = n_steps)

    return training_loop

model = classifier()
output_dir = 'model/'
output_dir_expand = os.path.expanduser(output_dir)
training_loop = train_model(model, train_task, eval_task, 100, output_dir_expand)


'''
    Part4：评估模型
'''


def compute_accuracy(preds, y, y_weights):
    """
    计算给定预测结果和真实标签的准确率
    Input:
        preds: 一个形状为 (dim_batch, output_dim) 的张量，表示模型的预测结果。对正负情感的概率预测，因此 output_dim 为 2。
        y: 一个形状为 (dim_batch, output_dim) 的张量，包含真实的标签。同样，对于二分类问题，output_dim 为 2。
        y_weights: 一个 numpy.ndarray，为每个样本分配一个权重。
    Output:
        accuracy: a float between 0-1
        weighted_num_correct (np.float32): Sum of the weighted correct predictions
        sum_weights (np.float32): Sum of the weights
    """
    # 判断每个样本的正类概率是否大于负类概率，得到一个布尔数组。
    # 比如preds的维度是一个(64,2)，preds[:, 1]表示取这个张量中所有行的第二个维度上的元素，preds[:, 0]则表示第一维度上的元素
    is_pos = preds[:, 1] > preds[:, 0]

    # 将布尔数组转换为整数数组，方便后续与真实标签进行比较。
    is_pos_int = is_pos.astype(np.int32)

    # 将预测结果与真实标签进行比较，得到一个表示每个样本是否预测正确的布尔数组。
    correct = is_pos_int == y

    # 计算所有样本的权重之和。
    sum_weights = np.sum(y_weights)

    # 转换正确预测为浮点型并加权
    correct_float = correct.astype(np.float32)

    # 计算加权正确预测总数和准确率
    weighted_correct_float = correct_float * y_weights
    weighted_num_correct = np.sum(weighted_correct_float)
    accuracy = weighted_num_correct / sum_weights

    return accuracy, weighted_num_correct, sum_weights


def test_model(generator, model):
    '''
    测试数据准确率
    Input:
        generator: an iterator instance that provides batches of inputs and targets
        model: a model instance
    Output:
        accuracy: float corresponding to the accuracy
    '''

    accuracy = 0.
    total_num_correct = 0
    total_num_pred = 0

    for batch in generator:
        inputs = batch[0]
        targets = batch[1]
        example_weight = batch[2]

        pred = model(inputs)

        batch_accuracy, batch_num_correct, batch_num_pred = compute_accuracy(pred, targets, example_weight)
        total_num_correct += batch_num_correct
        total_num_pred += batch_num_pred

    accuracy = total_num_correct / total_num_pred

    return accuracy

model = training_loop.eval_model
accuracy = test_model(test_generator(16), model)

print(f'The accuracy of your model on the validation set is {accuracy:.4f}', )


def predict(sentence):
    inputs = np.array(tweet_to_tensor(sentence, vocab_dict=Vocab))

    inputs = inputs[None, :]

    preds_probs = model(inputs)

    preds = int(preds_probs[0, 1] > preds_probs[0, 0])

    sentiment = "negative"
    if preds == 1:
        sentiment = 'positive'

    return preds, sentiment

sentence = "It's such a nice day, think i'll be taking Sid to Ramsgate fish and chips for lunch at Peter's fish factory and then the beach maybe"
tmp_pred, tmp_sentiment = predict(sentence)
print(f"The sentiment of the sentence \n***\n\"{sentence}\"\n***\nis {tmp_sentiment}.")

print()
# try a negative sentence
sentence = "I hated my day, it was the worst, I'm so sad."
tmp_pred, tmp_sentiment = predict(sentence)
print(f"The sentiment of the sentence \n***\n\"{sentence}\"\n***\nis {tmp_sentiment}.")