import trax
from trax import layers as tl
import numpy as np
import pandas as pd
from utils import get_params, get_vocab
import random as rnd
from trax.supervised import training

'''
    基于Trax框架的命名实体识别。
    本训练是有监督学习，训练集为句子，标签为句子中每个单词对应的命名实体类别。模型通过学习输入（句子）和输出（标签）之间的映射关系来进行预测。
    单词和标签的特征化（数值表示）是通过构建词汇表(vocab)和标签映射(tag_map)完成的。
    流程：数据处理、模型构建、前向传播、损失计算、后向传播、权重更新。
'''

'''
    Part1：数据处理
'''
data = pd.read_csv("ner_dataset.csv", encoding="ISO-8859-1")

# vocab：词汇表，key为单词，value为其对应的唯一索引（从0递增）。 tag_map：标签表，与词汇表类似，key为类别，比如B-geo，value为其对应的索引（从0递增）
vocab, tag_map = get_vocab('data/large/words.txt', 'data/large/tags.txt')

# 将句子和标签转换为索引。t_sentences：large训练集中每个句子的每个单词在词汇表中对应的索引，一个句子是一个列表。 t_labels：每个句子的每个单词对应类别的列表。
t_sentences, t_labels, t_size = get_params(vocab, tag_map, 'data/large/train/sentences.txt',
                                           'data/large/train/labels.txt')
# 与上列字段格式相同，为验证集数据
v_sentences, v_labels, v_size = get_params(vocab, tag_map, 'data/large/val/sentences.txt', 'data/large/val/labels.txt')
# 与上列字段格式相同，为测试集数据
test_sentences, test_labels, test_size = get_params(vocab, tag_map, 'data/large/test/sentences.txt',
                                                    'data/large/test/labels.txt')

g_vocab_size = len(vocab)


def data_generator(batch_size, x, y, pad, shuffle=False, verbose=False):
    '''
      Input:
        batch_size - 批量的量
        x - 句子列表，句子中的单词用整数表示，比如上面的t_sentences
        y - 每个句子的单词对应类别的列表。比如上面的t_labels
        shuffle - 是否随机打乱顺序
        pad - 填充符号的索引
        verbose - 是否打印信息
      Output:
        a tuple containing 2 elements:
        X - 矩阵，维度：(batch_size, max_len)
        Y - 矩阵，维度： (batch_size, max_len) ,X的标签
    '''

    ''' 第一步：将输入的x、y打乱顺序，并按照每个批次的数量（batch_size），存储到buffer_x和buffer_y中，记录最长的句子的长度。'''

    # 句子数量
    num_lines = len(x)

    # 创建一个下标列表
    lines_index = [*range(num_lines)]

    if shuffle:
        rnd.shuffle(lines_index)

    # 定位x,y
    index = 0
    while True:
        buffer_x = [0] * batch_size
        buffer_y = [0] * batch_size

        max_len = 0
        for i in range(batch_size):
            # 重置定位并打乱顺序
            if index >= num_lines:
                index = 0
                if shuffle:
                    rnd.shuffle(lines_index)

            # 根据下标存储已打乱乱序的数据集
            buffer_x[i] = x[lines_index[i]]
            buffer_y[i] = y[lines_index[i]]

            lenx = len(buffer_x[i])
            if lenx > max_len:
                max_len = lenx
            index += 1

        ''' 第二步：按(batch_size, max_len)的形状，初始化X、Y。'''

        # 初始化X、Y，使用pad创建X、Y的空数组
        X = np.full((batch_size, max_len), pad)
        Y = np.full((batch_size, max_len), pad)

        ''' 第三步：将buffer_x和buffer_y中的数据逐个存储到X、Y并输出。 '''
        # 从buffer中将数据拷贝到X、Y中
        for i in range(batch_size):
            x_i = buffer_x[i]
            y_i = buffer_y[i]

            for j in range(len(x_i)):
                X[i, j] = x_i[j]
                Y[i, j] = y_i[j]

        if verbose: print("index=", index)
        yield ((X, Y))


'''
    Part2：创建模型
'''


def NER(vocab_size=35181, d_model=50, tags=tag_map):
    '''
        命名实体识别模型
      Input:
        vocab_size - 词汇表size
        d_model - 嵌入量的维度
      Output:
        model - 一个trax model
    '''
    model = tl.Serial(
        # Embedding层将每个单词的索引转化为一个固定长度的向量（词嵌入），使得模型可以处理词语之间的语义关系
        tl.Embedding(vocab_size=vocab_size, d_feature=d_model),
        # LSTM层捕捉句子中单词之间的上下文关系，特别是长句子的依赖关系
        tl.LSTM(n_units=d_model),
        # 全连接层：将LSTM的输出映射到与标签数量相同的维度空间，生成每个单词对应的实体类别预测。
        tl.Dense(n_units=len(tags)),
        # LosSoftMax层：将输出的分数转化为概率分布，每个类别都有一个概率值，所有类别的概率和为1
        tl.LogSoftmax()

        # 流程： 输入的句子首先经过Embedding层，每个单词被转换为向量表示。接着，向量输入LSTM层，LSTM通过它的状态机制，
        # 逐步更新每个时间步的隐藏状态，并输入句子中每个单词的特征表示。 这些特征表示传递到全连接层，将特征映射到对应的类别数
        # （标签数），生成每个单词属于不同实体类别的分数，LogSoftmax对每个分数进行归一化。
    )
    return model


# model = NER()

'''
    Part3：训练模型
'''
rnd.seed(33)

batch_size = 64

# 创建训练数据生成器 trax.data.add_loss_weights 是一个用于为数据集添加损失权重的函数。
train_generator = trax.data.inputs.add_loss_weights(
    data_generator(batch_size, t_sentences, t_labels, vocab['<PAD>'], True),
    id_to_mask=vocab['<PAD>'])

# 创建校验数据生成器
eval_generator = trax.data.inputs.add_loss_weights(
    data_generator(batch_size, v_sentences, v_labels, vocab['<PAD>'], True),
    id_to_mask=vocab['<PAD>'])


def train_model(NER, train_generator, eval_generator, train_steps=1, output_dir='model'):
    '''
    训练模型
    Input:
        NER - 已经创建的模型
        train_generator - 训练样本数据生成器
        eval_generator - 评估样本数据生成器
        train_steps - 训练步骤数
        output_dir - 模型保存目录
    Output:
        training_loop - 一个有监督的 trax 训练循环。
    '''

    train_task = training.TrainTask(
        train_generator,
        loss_layer=tl.CrossEntropyLoss(), # 使用交叉熵损失函数
        optimizer=trax.optimizers.Adam(0.01),
    )

    eval_task = training.EvalTask(
        labeled_data=eval_generator,
        metrics=[tl.CrossEntropyLoss(), tl.Accuracy()],
        n_eval_batches=10
    )

    training_loop = training.Loop(
        NER,
        train_task,
        eval_tasks=eval_task,
        output_dir=output_dir)

    training_loop.run(n_steps=train_steps)
    return training_loop


# 此处直接使用课程组提供的model，当然，用自己生成的model训练也可以。
# train_steps = 100
# training_loop = train_model(NER(), train_generator, eval_generator, train_steps)

model = NER()
model.init(trax.shapes.ShapeDtype((1, 1), dtype=np.int32))
# 加载已经预训练的model
model.init_from_file('model/model.pkl.gz', weights_only=True)

'''
    Part4：计算准确率
'''


def evaluate_prediction(pred, labels, pad):
    """
    Inputs:
        pred: 对测试数据进行训练后得到预测数据，类型为矩阵，形状：(样本批次数量, 批次中最大句子长度, 类别数量)
        labels: (样本批次数量, 批次中最大句子长度)
        pad: pad字符的数字表示
    Outputs:
        accuracy: float
    """
    # 取预测矩阵的第三个维度（即类别）的最大的下标，与lables比较
    outputs = np.argmax(pred, axis=2)
    print("outputs shape:", outputs.shape)
    mask = np.array(pad != labels)
    accuracy = np.sum(outputs == labels) / float(np.sum(mask))
    return accuracy


x, y = next(data_generator(len(test_sentences), test_sentences, test_labels, vocab['<PAD>']))
accuracy = evaluate_prediction(model(x), y, vocab['<PAD>'])
print("accuracy: ", accuracy)

'''
    Part5：测试自己的句子
'''


def predict(sentence, model, vocab, tag_map):
    s = [vocab[token] if token in vocab else vocab['UNK'] for token in sentence.split(' ')]
    batch_data = np.ones((1, len(s)))
    batch_data[0][:] = s
    sentence = np.array(batch_data).astype(int)
    output = model(sentence)
    outputs = np.argmax(output, axis=2)
    labels = list(tag_map.keys())
    pred = []
    for i in range(len(outputs[0])):
        idx = outputs[0][i]
        pred_label = labels[idx]
        pred.append(pred_label)
    return pred


sentence = "Peter Navarro, the White House director of trade and manufacturing policy of U.S, said in an interview on Sunday morning that the White House was working to prepare for the possibility of a second wave of the coronavirus in the fall, though he said it wouldn’t necessarily come"
s = [vocab[token] if token in vocab else vocab['UNK'] for token in sentence.split(' ')]
predictions = predict(sentence, model, vocab, tag_map)
for x, y in zip(sentence.split(' '), predictions):
    if y != 'O':
        print(x, y)
