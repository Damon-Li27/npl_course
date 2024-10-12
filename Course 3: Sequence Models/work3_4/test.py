import os
import nltk
import trax
from trax import layers as tl
from trax.supervised import training
from trax.fastmath import numpy as fastnp
import numpy as np
import pandas as pd
import random as rnd
from collections import defaultdict
from functools import partial

rnd.seed(34)

'''
    基于trax框架编码的孪生神经网络的模型，用于检测成对问题是否相似。这是一个有监督学习任务。
    模型通过训练数据（问题对及其是否重复的标签）进行训练，训练过程中使用了三元组损失函数（Triplet Loss），
    该损失函数帮助模型学习区分重复问题和不重复问题。调整权重以最小化损失函数。训练完成后，
    模型能够根据新输入的问题对，预测它们是否重复。
'''


'''
    Part1：引入数据。
    只选用了相同问题的问题对作为训练数据，例如，[q1a, q2a, q3a, ...]和[q1b, q2b,q3b, ...]， (q1a,q1b) 是重复的，(q1a,q2i) i!=a 是不重复的。
'''

''' 第一步：读取csv文件数据，按指定数量分为训练集和测试集 '''

data = pd.read_csv("questions.csv")
N = len(data)
print('Number of question pairs: ', N)

N_train = 300000
N_test = 10 * 1024
data_train = data[:N_train]
data_test = data[N_train:N_train + N_test]

# 删除data，释放内存，后续使用data_train和data_test
del (data)

''' 第二步：从数据中获取重复问题的问题1和问题2，训练数据和测试数据分别获取。 '''

td_index = (data_train['is_duplicate'] == 1).to_numpy()
# 获取td_index中为true的所有下标，生成列表。列表中的每个下标表示，该下标所指向的那行数据中的两个问题是重复的。
td_index = [i for i, x in enumerate(td_index) if x]

# 获取所有重复问题的问题1和问题2的内容
Q1_train_words = np.array(data_train['question1'][td_index])
Q2_train_words = np.array(data_train['question2'][td_index])

# 获取测试集的所有问题1和问题2
Q1_test_words = np.array(data_test['question1'])
Q2_test_words = np.array(data_test['question2'])
y_test = np.array(data_test['is_duplicate'])

# empty_like用于创建一个与给定数组具有相同形状和数据类型的空数组。
Q1_train = np.empty_like(Q1_train_words)
Q2_train = np.empty_like(Q2_train_words)

Q1_test = np.empty_like(Q1_test_words)
Q2_test = np.empty_like(Q2_test_words)

''' 第三步：创建包含所有训练数据的词汇表，key单词，value为数值。'''

# 创建词汇表，defaultdict对象，这个对象在访问不存在的键时会返回默认值0
vocab = defaultdict(lambda: 0)
vocab['<PAD>'] = 1

for idx in range(len(Q1_train_words)):
    Q1_train[idx] = nltk.word_tokenize(Q1_train_words[idx])
    Q2_train[idx] = nltk.word_tokenize(Q2_train_words[idx])
    q = Q1_train[idx] + Q2_train[idx]
    for word in q:
        if word not in vocab:
            vocab[word] = len(vocab) + 1

for idx in range(len(Q1_test_words)):
    Q1_test[idx] = nltk.word_tokenize(Q1_test_words[idx])
    Q2_test[idx] = nltk.word_tokenize(Q2_test_words[idx])

''' 第四步：根据词汇表，将Q1，Q2的所有单词转换成数值，即将每个句子转换成了向量'''
for i in range(len(Q1_train)):
    Q1_train[i] = [vocab[word] for word in Q1_train[i]]
    Q2_train[i] = [vocab[word] for word in Q2_train[i]]

for i in range(len(Q1_test)):
    Q1_test[i] = [vocab[word] for word in Q1_test[i]]
    Q2_test[i] = [vocab[word] for word in Q2_test[i]]

''' 第五步：切割训练数据，前80%为训练集，后20%为验证集'''
cut_off = int(len(Q1_train) * .8)
train_Q1, train_Q2 = Q1_train[:cut_off], Q2_train[:cut_off]
val_Q1, val_Q2 = Q1_train[cut_off:], Q2_train[cut_off:]

''' 第六步：创建数据生成器：打乱顺序，取长度最长的句子作为最大长度并向上取整到2的幂，然后对batches批量填充'''
def data_generator(Q1, Q2, batch_size, pad=1, shuffle=True):
    """
        数据生成器
    Args:
        Q1 (list): 提转换成向量的Q1问题列表
        Q2 (list): 提转换成向量的Q2问题列表
        batch_size (int): 每个批次的元素数量
        pad (int, optional): 填充字符，默认1
        shuffle (bool, optional): 是否随机打乱顺序
    Yields:
        tuple:  格式：(input1, input2)， 类型： (numpy.ndarray, numpy.ndarray)
        NOTE: input1: 用于模型输入，[q1a, q2a, q3a, ...] 比如 (q1a,q1b) 是重复的
              input2: 用于模型输出 [q1b, q2b,q3b, ...] 比如. (q1a,q2i) i!=a 是不重复的
    """

    input1 = []
    input2 = []
    idx = 0
    len_q = len(Q1)
    question_indexes = [*range(len_q)]

    if shuffle:
        rnd.shuffle(question_indexes)

    while True:
        # 当数据遍历完后，重置并打乱顺序
        if idx >= len_q:
            idx = 0
            if shuffle:
                rnd.shuffle(question_indexes)

        q1 = Q1[question_indexes[idx]]
        q2 = Q2[question_indexes[idx]]
        idx += 1
        input1.append(q1)
        input2.append(q2)
        if len(input1) == batch_size:
            # 从input1和input2中选长度最大那个句子，作为句子的最大长度
            max_len = max(max(len(x) for x in input1), max(len(x) for x in input2))
            # 为了便于计算，将max_len向上取整为2的幂（一般使用2的指数条数据作为一个批次）
            max_len = 2 ** int(np.ceil(np.log2(max_len)))
            b1 = []
            b2 = []
            for q1, q2 in zip(input1, input2):
                # 填充pad
                q1 = q1 + [pad] * (max_len - len(q1))
                q2 = q2 + [pad] * (max_len - len(q2))

                b1.append(q1)
                b2.append(q2)
            yield np.array(b1), np.array(b2)
            # 重置batches
            input1, input2 = [], []


'''
    Part2：定义孪生神经网络和损失函数
'''


def Siamese(vocab_size=len(vocab), d_model=128, mode='train'):
    """返回一个孪生神经网络

    Args:
        vocab_size (int, optional): 词汇表长度
        d_model (int, optional): model的深度，默认128
        mode (str, optional): 'train', 'eval' or 'predict'，默认'train'.

    Returns:
        使用trax生成的Siamese网络
    """

   # 对输入x进行归一化。每个向量x除以它的L2范数,这种归一化将所有向量的L2范数归一到1，
    # 因此，两个向量的点积（即上面的scores矩阵的元素）会被限制在[-1, 1]范围内。
    def normalize(x):
        return x / fastnp.sqrt(fastnp.sum(x * x, axis=-1, keepdims=True))

    q_processor = tl.Serial(
        tl.Embedding(vocab_size=vocab_size, d_feature=d_model),
        tl.LSTM(n_units=d_model),
        tl.Mean(axis=1),  # 计算张量均值
        # tl.Fn 用于创建一个自定义的函数层，第一个参数可以用于在模型结构中标识这个层，方便调试和理解模型架构。
        # 第二个参数是一个匿名函数，接受一个输入x，并调用之前定义的 normalize 函数对其进行处理
        tl.Fn('Normalization', lambda x: normalize(x))
    )

    # tl.Parallel 是一个用于并行组合多个层的容器。允许同时应用多个不同的层或操作于相同的输入数据，
    # 然后将这些层的输出组合在一起。Siamese网络是两个相同的层。
    model = tl.Parallel(q_processor, q_processor)
    return model


def TripletLossFn(v1, v2, margin=0.25):
    """自定义损失函数

    Args:
        v1 (numpy.ndarray): 关联Q1，形状为： (batch_size, model_dimension)
        v2 (numpy.ndarray): 关联Q2，形状为： (batch_size, model_dimension)
        margin (float, optional): 期望边界，即损失函数中的α. 默认 0.25.

    Returns:
        jax.interpreters.xla.DeviceArray: Triplet Loss.
    """

    # 矩阵点乘获得评分矩阵，形状为：（batch_size，batch_size）
    scores = fastnp.dot(v1, v2.T)
    # 样本数量
    batch_size = len(scores)
    # 抓取scores的对角线元素，表示正样本与锚点的点积值
    positive = fastnp.diagonal(scores)
    ''' 第一步：算得最接近负样本损失函数  '''
    # 将对角线（正样本相似度）置为负数，然后按行取最大值，即找到每个样本与其他样本中最接近的负样本相似度。
    negative_without_positive = scores - 2.0 * fastnp.eye(batch_size)
    closest_negative = negative_without_positive.max(axis=1)
    # 损失函数公式1（最接近负样本损失函数）
    triplet_loss1 = fastnp.maximum(0.0, margin - positive + closest_negative)
    ''' 第二步：算得负样本均值损失函数  '''
    # 将对角线元素置为0，得到只包含负样本相似度的矩阵。（对应元素相乘）
    negative_zero_on_duplicate = scores * (1.0 - fastnp.eye(batch_size))
    # 按行求和，然后除以batch_size - 1，得到每个样本与其他样本的平均负样本相似度。
    mean_negative = np.sum(negative_zero_on_duplicate, axis=1) / (batch_size - 1)
    # 损失函数公式2（负样本平均值损失函数）
    triplet_loss2 = fastnp.maximum(0.0, margin - positive + mean_negative)

    ''' 第三步：两个损失项相加，然后取平均值，得到最终的三元组损失  '''
    triplet_loss = fastnp.mean(triplet_loss1 + triplet_loss2)

    return triplet_loss


'''
    Part3：训练模型
'''


def TripletLoss(margin=0.25):
    # functools.partial可以用来创建一个新的函数，这个新函数是对已有函数的部分参数进行了固定。
    triplet_loss_fn = partial(TripletLossFn, margin=margin)
    return tl.Fn('TripletLoss', triplet_loss_fn)


# trax.lr.warmup_and_rsqrt_decay 是一种学习率调度策略。第一个参数：控制 warmup 阶段的长度，即学习率从初始值增加到最大值所需的训练步数。
# 第二个参数：设定 warmup 阶段结束后的最大学习率。
lr_schedule = trax.lr.warmup_and_rsqrt_decay(400, 0.01)

batch_size = 256
train_generator = data_generator(train_Q1, train_Q2, batch_size, vocab['<PAD>'])
val_generator = data_generator(val_Q1, val_Q2, batch_size, vocab['<PAD>'])


def train_model(Siamese, TripletLoss, lr_schedule, train_generator=train_generator, val_generator=val_generator,
                output_dir='model/'):
    """训练Siamese网络模型

    Args:
        Siamese (function): 上面定义的Siamese函数
        TripletLoss (function): 定义的损失函数
        lr_schedule (function): lr_schedule
        train_generator (generator, optional): 训练集生成器
        val_generator (generator, optional): 验证集生成器
        output_dir (str, optional): 输出目录

    Returns:
        trax.supervised.training.Loop: 模型的训练loop。
    """
    output_dir = os.path.expanduser(output_dir)

    train_task = training.TrainTask(
        labeled_data=train_generator,
        loss_layer=TripletLoss(),
        optimizer=trax.optimizers.Adam(0.01),
        lr_schedule=lr_schedule,
    )

    eval_task = training.EvalTask(
        labeled_data=val_generator,  # Use generator (val)
        metrics=[TripletLoss()],  # Use triplet loss. Don't forget to instantiate this object
    )

    training_loop = training.Loop(Siamese(),
                                  train_task,
                                  eval_tasks=eval_task,
                                  output_dir=output_dir)

    return training_loop

# 已经训练之后就可以注释
# train_steps = 200
# training_loop = train_model(Siamese, TripletLoss, lr_schedule)
# training_loop.run(train_steps)

'''
    上面的流程基本上如此，但在训练过程中损失值并未减小，代码应该没有问题。以下代码用官方model训练
    
    Part4：评估模型
'''

model = Siamese()
model.init_from_file('model.pkl.gz')


def classify(test_Q1, test_Q2, y, threshold, model, vocab, data_generator=data_generator, batch_size=64):
    """
        测试模型准确率
    Args:
        test_Q1 (numpy.ndarray): Q1问题的向量矩阵
        test_Q2 (numpy.ndarray): Q2问题向量矩阵
        y (numpy.ndarray): 实际的标签值矩阵
        threshold (float): 阈值
        model (trax.layers.combinators.Parallel): Siamese模型.
        vocab (collections.defaultdict): 词汇表
        data_generator (function): 数据生成器
        batch_size (int, optional): 批量大小，默认 64.

    Returns:
        float: 模型准确率
    """
    accuracy = 0
    for i in range(0, len(test_Q1), batch_size):
        # q1、q2的形状：(512, 64)，每次生成batch_size数量的样本
        q1, q2 = next(data_generator(test_Q1[i:i + batch_size], test_Q2[i:i + batch_size], batch_size, vocab['<PAD>'],
                                     shuffle=False))
        # y_test的形状：(512,)
        y_test = y[i:i + batch_size]
        # 将q1,q2作为输入训练
        v1, v2 = model((q1, q2))

        for j in range(batch_size):
            d = np.dot(v1[j], v2[j].T)
            res = 1 if d > threshold else 0
            accuracy += (y_test[j] == res)
    accuracy = accuracy / len(test_Q1)

    return accuracy

accuracy = classify(Q1_test, Q2_test, y_test, 0.7, model, vocab, batch_size=512)
print("Accuracy", accuracy)



'''
    Part5：测试自己的questions
'''


def predict(question1, question2, threshold, model, vocab, data_generator=data_generator, verbose=False):
    """
        测试自己的数据
    Args:
        question1 (str): 问题1
        question2 (str): 问题2
        threshold (float): 阀值
        model (trax.layers.combinators.Parallel): Siamese模型.
        vocab (collections.defaultdict): 词汇表.
        data_generator (function): 数据生成器
        verbose (bool, optional): 是否打印

    Returns:
        bool: 如果问题相同，返回True，否则False
    """
    q1 = nltk.word_tokenize(question1)
    q2 = nltk.word_tokenize(question2)
    Q1, Q2 = [], []
    for word in q1:
        Q1 += [vocab[word]]
    for word in q2:
        Q2 += [vocab[word]]


    Q1, Q2 = next(data_generator([Q1], [Q2], 1, vocab['<PAD>']))

    v1, v2 = model((Q1,Q2))

    d = np.dot(v1[0], v2[0].T)

    res = d > threshold

    if (verbose):
        print("Q1  = ", Q1, "\nQ2  = ", Q2)
        print("d   = ", d)
        print("res = ", res)

    return res


# %%
# Feel free to try with your own questions
question1 = "When will I see you?"
question2 = "When can I see you again?"
# 1 means it is duplicated, 0 otherwise
predict(question1, question2, 0.7, model, vocab, verbose=True)

question3 = "Do they enjoy eating the dessert?"
question4 = "Do they like hiking in the desert?"
# 1 means it is duplicated, 0 otherwise
predict(question3 , question4, 0.7, model, vocab, verbose=True)
