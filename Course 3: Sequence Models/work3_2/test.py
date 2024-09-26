import itertools
import os
import random as rnd
import numpy
import trax
import trax.fastmath.numpy as np
from trax import layers as tl
from trax.supervised import training
rnd.seed(32)

'''
    使用 Trax 库实现基于 GRU（门控循环单元）的语言模型：学习输入文本的模式和规律，以便能够生成新的文本或对给定的文本进行预测。
    在这个语言模型中，输入的文本序列同时也作为标签，因为模型的任务是预测下一个字符或序列，所以目标就是输入序列本身。
    注意：每次运行需要删除model下的文件。
'''

'''
    Part1: 数据加载
'''
dirname = 'data/'
lines = []
# 从data目录下加载数据
for filename in os.listdir(dirname):
    with open(os.path.join(dirname, filename)) as files:
        for line in files:
            pure_line = line.strip()
            if pure_line:
                lines.append(pure_line)
# %%
n_lines = len(lines)

eval_lines = lines[-1000:]  # 去最后1000行作验证集
lines = lines[:-1000]  # 获取从第一个元素开始直到倒数第 1000 个元素，作为训练集


def line_to_tensor(line, EOS_int=1):
    """
    文本转为向量（unicode编码）
    Args:
        line (str):一行文本
        EOS_int (int, optional): 句子结尾符号，默认1

    Returns:
        list: 文本的unicode编码组成的列表
    """
    tensor = []

    for c in line:
        # 转为unicode编码
        c_int = ord(c)
        tensor.append(c_int)

    tensor.append(EOS_int)
    return tensor


def data_generator(batch_size, max_length, data_lines, line_to_tensor=line_to_tensor, shuffle=True):
    """
        生成批量的向量数据，每个line长度都为max_length，长度不足的用0填充。用mask_np_arr作为掩码，判断哪些数据有效。
    Args:
        batch_size (int): 每个批量的数据个数（长度）
        max_length (int): 最大长度
        NOTE: 最大长度包含句子结束符号
        data_lines (list): 数据行
        line_to_tensor (function, optional): 一行文本转为向量的函数
        shuffle (bool, optional): 为True是随机打乱顺序

    Yields:
        tuple: two copies of the batch (jax.interpreters.xla.DeviceArray) and mask (jax.interpreters.xla.DeviceArray).
        NOTE: jax.interpreters.xla.DeviceArray is trax's version of numpy.ndarray
    """
    index = 0
    cur_batch = []
    num_lines = len(data_lines)

    # 数据行的索引列表，便于打乱顺序
    # 星号（*）在这里用于解包（unpacking）可迭代对象。如果没有这个星号，结果将是一个包含一个range对象的列表，而不是包含具体数字的列表。
    lines_index = [*range(num_lines)]

    if shuffle:
        rnd.shuffle(lines_index)

    while True:  # 无限循环，外部控制输出的次数
        if index >= num_lines:  # 遍历完一轮后，重置index
            index = 0
            if shuffle:
                rnd.shuffle(lines_index)

        # 获取数据，直到cur_batch的长度等于要求的batch_size；长度大于所设的max_length的句子，就跳过
        line = data_lines[lines_index[index]]
        if len(line) < max_length:
            cur_batch.append(line)
        index += 1

        # 转向量
        if len(cur_batch) == batch_size:
            batch = []
            mask = []

            for li in cur_batch:
                tensor = line_to_tensor(li)
                pad = [0] * (max_length - len(tensor))
                tensor_pad = tensor + pad
                batch.append(tensor_pad)

                # 在自然语言处理任务中，输入的文本序列往往具有不同的长度。为了能够以批量的方式处理这些序列，通常需要将它们填充到相同的长度。
                # 但是，在计算损失或进行其他操作时，填充的部分不应该参与计算，因为它们并不是真正的有效数据。
                # mask_np_arr就是用来标识哪些位置是有效数据，哪些位置是填充的数据。
                example_mask = [1 if element > 0 else 0 for element in tensor_pad]
                mask.append(example_mask)

            batch_np_arr = np.array(batch)
            mask_np_arr = np.array(mask)

            # 标签与训练集的数据相同
            yield batch_np_arr, batch_np_arr, mask_np_arr
            cur_batch = []


'''
    Part2: 定义GRU模型
'''


def GRULM(vocab_size=256, d_model=512, n_layers=2, mode='train'):
    """ 返回一个GRU模型.

    Args:
        vocab_size (int, optional): 词汇表大小，默认256
        d_model (int, optional): 嵌入深度（GRU单元数），默认512
        n_layers (int, optional): GRU层数，默认2层
        mode (str, optional): 'train', 'eval' 或 'predict',
    Returns:
        trax.layers.combinators.Serial: A GRU language model as a layer that maps from a tensor of tokens to activations over a vocab set.
    """
    model = tl.Serial(
        tl.ShiftRight(mode=mode),  # 右移操作，右移在某些情况下可以用于引入位置信息或进行特定的数据预处理
        tl.Embedding(vocab_size=vocab_size, d_feature=d_model),  # 嵌入层。它将输入数据中的词汇索引（整数）转换为密集的向量表示。
        [tl.GRU(n_units=d_model) for _ in range(n_layers)],  # 由多个门控循环单元（GRU）层组成的列表。创建n_layers个相同的 GRU 层，每个 GRU 层有n_units个单元。
        tl.Dense(n_units=vocab_size),  # 全连接层
        tl.LogSoftmax()
    )
    return model

'''
    Part3: 训练
'''
def n_used_lines(lines, max_length):
    '''
    计算在整个数据集中有多少行符合我们的句子最大长度标准
    Args:
    lines: 文本所有行
    max_length - 一个行的最大长度
    output_dir - 输出目录
    Return:
    number of efective examples
    '''
    n_lines = 0
    for l in lines:
        if len(l) <= max_length:
            n_lines += 1
    return n_lines


def train_model(model, data_generator, batch_size=32, max_length=64, lines=lines, eval_lines=eval_lines, n_steps=1,
                output_dir='model/'):
    """
    训练模型

    Returns:
        trax.supervised.training.Loop: Training loop for the model.
    """

    bare_train_generator = data_generator(batch_size, max_length, data_lines=lines)
    infinite_train_generator = itertools.cycle(bare_train_generator)

    bare_eval_generator = data_generator(batch_size, max_length, data_lines=eval_lines)
    infinite_eval_generator = itertools.cycle(bare_eval_generator)

    train_task = training.TrainTask(
        labeled_data=infinite_train_generator,
        loss_layer=tl.CrossEntropyLoss(),
        optimizer=trax.optimizers.Adam(0.0005)
    )

    eval_task = training.EvalTask(
        labeled_data=infinite_eval_generator,
        metrics=[tl.CrossEntropyLoss(), tl.Accuracy()],
        n_eval_batches=3
    )
    training_loop = training.Loop(model,
                                  train_task,
                                  eval_tasks=[eval_task],
                                  output_dir=output_dir)
    training_loop.run(n_steps=n_steps)
    return training_loop


training_loop = train_model(GRULM(), data_generator)

'''
    Part4： 模型评估
'''


def test_model(preds, target):
    """
        测试模型
    Args:
        preds (jax.interpreters.xla.DeviceArray): Predictions of a list of batches of tensors corresponding to lines of text.
        target (jax.interpreters.xla.DeviceArray): Actual list of batches of tensors corresponding to lines of text.

    Returns:
        float: log_perplexity of the model.
    """
    # 计算总对数概率total_log_ppx，通过将预测结果preds与目标target经过tl.one_hot处理后的结果相乘，然后在最后一个维度上求和。
    total_log_ppx = np.sum(preds * tl.one_hot(target, preds.shape[-1]),
                           axis=-1)

    non_pad = 1.0 - np.equal(target, 0)
    ppx = total_log_ppx * non_pad

    log_ppx = np.sum(ppx) / np.sum(non_pad)

    return -log_ppx
# 创建一个新的 GRU 模型实例。从指定的文件中加载模型参数初始化模型。
model = GRULM()
model.init_from_file('model/model.pkl.gz')

'''
    Part5： 生成自己的语言模型
'''


def gumbel_sample(log_probs, temperature=1.0):
    """
    函数实现了从一个类别分布中进行 Gumbel 采样。Gumbel 采样是一种用于从离散分布中采样的技术
    :param log_probs: 表示每个类别对应的对数概率
    :param temperature: 一个浮点数，控制采样的随机性，温度越高，采样结果越随机；温度越低，采样结果越接近概率最大的类别。
    :return:
    """
    u = numpy.random.uniform(low=1e-6, high=1.0 - 1e-6, size=log_probs.shape)
    g = -np.log(-np.log(u))
    return np.argmax(log_probs + g * temperature, axis=-1)


def predict(num_chars, prefix):
    '''
    使用已训练的模型进行预测，并通过 Gumbel 采样选择下一个字符，直到生成指定数量的字符或遇到结束符号
    :param num_chars:
    :param prefix:
    :return:
    '''
    inp = [ord(c) for c in prefix]
    result = [c for c in prefix]
    max_len = len(prefix) + num_chars
    for _ in range(num_chars):
        cur_inp = np.array(inp + [0] * (max_len - len(inp)))
        outp = model(cur_inp[None, :])  # Add batch dim.
        next_char = gumbel_sample(outp[0, len(inp)])
        inp += [int(next_char)]

        if inp[-1] == 1:
            break  # EOS
        result.append(chr(int(next_char)))

    return "".join(result)


print(predict(32, ""))