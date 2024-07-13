import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import os
import tensorflow as tf

# tensorflow一直无法调用GPU，网上说加上下面这段可以解决，但在我的电脑上没有作用
# physical_devices = tf.config.list_physical_devices('GPU')
# for device in physical_devices:
#     tf.config.experimental.set_memory_growth(device, True)

from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.layers import Embedding, Bidirectional, LSTM, Masking, Dense, Input, TimeDistributed, Activation, Lambda, \
    Dropout
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from tqdm import tqdm
import keras


# 超参
BiRNN_UNITS = 200
BATCH_SIZE = 16  # 、64
EMBED_DIM = 300
EPOCHS = 2 # 因为没法调用GPU所以次数较小，有GPU的情况下可以增大到10次的数量级
optimizer = 'Adam' #可以切换优化器
# SGD = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
# RMSprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
NAME = 'cityu' #在此更换数据集

modelpath = os.path.join('Save model/' + NAME + '_model.h5')
datapath = os.path.join('icwb2-data/training/' + NAME + '_training.utf8')
testpath = os.path.join('icwb2-data/gold/' + NAME + '_test_gold.utf8')
savepath_txt = os.path.join('test_result/ML_with_nn/' + NAME + "_BiLSTM_crf_seg.txt")
savepath_utf8 = os.path.join('test_result/ML_with_nn/' + NAME + "_BiLSTM_crf_seg.utf8")


with open(datapath, 'r', encoding='utf8') as file:
    lines = file.readlines()
    train_set = pd.DataFrame([line.strip().split(',') for line in lines])
file.close()
with open(testpath, 'r', encoding='utf8') as file:
    lines = file.readlines()
    test_set = pd.DataFrame([line.strip().split(',') for line in lines])
file.close()


# 生成字序列
def get_char(sentence):
    char_list = []
    sentence = ''.join(sentence.split('  '))
    for i in sentence:
        char_list.append(i)
    return char_list


# 生成标注序列
def get_label(sentence):
    result = []
    word_list = sentence.split('  ')
    for i in range(len(word_list)):
        if len(word_list[i]) == 1:
            result.append('S')
        elif len(word_list[i]) == 2:
            result.append('B')
            result.append('E')
        else:
            temp = len(word_list[i]) - 2
            result.append('B')
            result.extend('M' * temp)
            result.append('E')
    return result


def read_file(file):
    char, content, label = [], [], []
    maxlen = 0

    for i in range(len(file)):
        line = file.loc[i, 0]
        line = line.strip('\n')
        line = line.strip(' ')
        char_list = get_char(line)
        label_list = get_label(line)
        maxlen = max(maxlen, len(char_list))
        if len(char_list) != len(label_list):
            continue  # 删掉可能有问题的样本
        char.extend(char_list)
        content.append(char_list)
        label.append(label_list)
    return char, content, label, maxlen  # word是单列表，content和label是双层列表


# 预处理: padding
def process_data(char_list, label_list, vocab, chunk_tags, MAXLEN):
    vocab2idx = {char: idx for idx, char in enumerate(vocab)}
    # 注意 <UNK>
    x = [[vocab2idx.get(char, 1) for char in s] for s in char_list]
    y_chunk = [[chunk_tags.index(label) for label in s] for s in label_list]

    # padding
    x = pad_sequences(x, maxlen=MAXLEN, value=0)
    y_chunk = pad_sequences(y_chunk, maxlen=MAXLEN, value=-1)

    # one_hot编码:
    y_chunk = to_categorical(y_chunk, len(chunk_tags))
    return x, y_chunk


# 加载数据
def load_data():
    chunk_tags = ['S', 'B', 'M', 'E']
    train_char, train_content, train_label, _ = read_file(train_set)
    test_char, test_content, test_label, maxlen = read_file(test_set)

    vocab = list(set(train_char + test_char))  # 合并，构成大词表
    special_chars = ['<PAD>', '<UNK>']  # 特殊词表示：PAD表示padding，UNK表示词表中没有
    vocab = special_chars + vocab

    # padding
    print('maxlen is %d' % maxlen)
    train_x, train_y = process_data(train_content, train_label, vocab, chunk_tags, maxlen)
    test_x, test_y = process_data(test_content, test_label, vocab, chunk_tags, maxlen)
    return train_x, train_y, test_x, test_y, vocab, chunk_tags, maxlen, test_content

train_x, train_y, test_x, test_y, vocab, chunk_tags, maxlen, test_content = load_data()

# 开源的中文词向量集，详见https://github.com/Embedding/Chinese-Word-Vectors
word2vec_model_path = 'sgns.context.word-character.char1-1.dynwin5.thr10.neg5.dim300.iter5.bz2'
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=False, unicode_errors='ignore')


def make_embeddings_matrix(word2vec_model, vocab):
    char2vec_dict = {}  # 字对词向量
    # vocab2idx = {char: idx for idx, char in enumerate(vocab)}
    for char, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
        char2vec_dict[char] = vector
    embeddings_matrix = np.zeros((len(vocab), EMBED_DIM))
    for i in tqdm(range(2, len(vocab))):
        char = vocab[i]
        if char in char2vec_dict.keys():  # 如果char在词向量列表中，更新权重；否则，赋值为全0
            char_vector = char2vec_dict[char]
            embeddings_matrix[i] = char_vector
    return embeddings_matrix

embeddings_matrix = make_embeddings_matrix(word2vec_model, vocab)

# 构造模型
# 输入层
inputs = Input(shape=(maxlen,), dtype='int32')
# 遮掩层
x = Masking(mask_value=0)(inputs)
# 嵌入层
x = Embedding(len(vocab), EMBED_DIM, weights=[embeddings_matrix], input_length=maxlen, trainable=True)(x)
# 双向LSTM层
x = Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True))(x)
# Dropout: 增加随机性
x = Dropout(0.5)(x)
# 全连接层
x = TimeDistributed(Dense(len(chunk_tags)))(x)
# 输出层，即crf层
outputs = CRF(len(chunk_tags))(x)

# print("Device name:", tensorflow.test.is_gpu_available())
model = Model(inputs = inputs, outputs = outputs)
# 打印模型参数
model.summary()

model.compile(optimizer=optimizer, loss=crf_loss, metrics=[crf_viterbi_accuracy])

# 训练
model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_split=0.1)
score = model.evaluate(test_x, test_y, batch_size=BATCH_SIZE)
print(score)
# 保存模型
model.save(modelpath)


# 测试
test_predict = model.predict(test_x) #test_x已经去掉空格，因此不需要从test.utf8从新导入
test_predict = [[np.argmax(char) for char in sample] for sample in test_predict]
test_predict_tag = [[chunk_tags[i] for i in sample] for sample in test_predict]  # 获得预测标签


test_result = []
vocab2idx = {char: idx for idx, char in enumerate(vocab)}
_, test_content, _, _ = read_file(test_set)
# 生成预测序列
for i in range(len(test_predict)):
    sentence = ''
    s_len = len(test_content[i])
    sample = test_predict_tag[i]
    for j in range(s_len):
        idx = len(sample) - s_len + j
        if sample[idx] == 'B' or sample[idx] == 'M' or j == s_len - 1:
            sentence = sentence + test_content[i][j]
        else:
            sentence = sentence + test_content[i][j]
            sentence = sentence + '  '
    test_result.append(sentence)


file = open(savepath_txt, 'w', encoding='utf-8-sig')
file1 = open(savepath_utf8, 'w', encoding='utf-8-sig')
for i in range(len(test_result)):
    file.write(test_result[i])
    file1.write(test_result[i])
    file.write('\n')
    file1.write('\n')
file.close()
file1.close()
