from __future__ import print_function
import numpy as np
import random
import jieba
import json

empty_vector = []
for i in range(0, 100):
    empty_vector.append(float(0.0))
onevector = []
for i in range(0, 10):
    onevector.append(float(1))
zerovector = []
for i in range(0, 10):
    zerovector.append(float(0))

# train_data="insuranceQA/train"
# test_data="insuranceQA/test1"
train_data="insuranceQA/train_data_allright_small"
test_data="insuranceQA/test_data_50"
vector_data="insuranceQA/vectors.nobin"

# 根据数据的每一个词语编号，写入一个dict
# 0表示未知UNKNOWN
def build_vocab():
    code = int(0)
    vocab = {}
    vocab['UNKNOWN'] = code
    code += 1

    for line in open(train_data,encoding='utf-8'):
        items = line.strip().split(' ')
        for i in range(2, 4):
            if len(items) < 4: continue
            words = items[i].split('_')
            for word in words:
                if not word in vocab:
                    vocab[word] = code
                    code += 1
    for line in open(test_data,encoding='utf-8'):
        items = line.strip().split(' ')
        for i in range(2, 4):
            if len(items)<4:continue
            words = items[i].split('_')
            for word in words:
                if not word in vocab:
                    vocab[word] = code
                    code += 1
    return vocab


# 随机拿一个qa对
def rand_qa(qalist):
    index = random.randint(0, len(qalist) - 1)
    return qalist[index]


# 读入答句序列
def read_alist():
    alist = []
    for line in open(train_data,encoding='utf-8'):
        items = line.strip().split(' ')
        alist.append(items[3])
    print('read_alist done ......')
    return alist


def vocab_plus_overlap(vectors, sent, over, size):
    global onevector
    global zerovector
    oldict = {}
    words = over.split('_')
    if len(words) < size:
        size = len(words)
    for i in range(0, size):
        if words[i] == '<a>':
            continue
        oldict[words[i]] = '#'
    matrix = []
    words = sent.split('_')
    if len(words) < size:
        size = len(words)
    for i in range(0, size):
        vec = read_vector(vectors, words[i])
        newvec = vec.copy()
        #if words[i] in oldict:
        #    newvec += onevector
        #else:
        #    newvec += zerovector
        matrix.append(newvec)
    return matrix

# 加载词向量到词典里
# 每个词一百维
def load_vectors():
    vectors = {}
    for line in open(vector_data,encoding='utf-8'):
        items = line.strip().split(' ')
        if (len(items) < 101):
            continue
        vec = []
        for i in range(1, 101):
            vec.append(float(items[i]))
        vectors[items[0]] = vec
    return vectors

# 根据词读它对应的向量
def read_vector(vectors, word):
    global empty_vector
    if word in vectors:
        return vectors[word]
    else:
        return empty_vector
        #return vectors['</s>']


def load_test_and_vectors():
    testList = []
    for line in open(test_data,encoding='utf-8'):
        if len(line.strip().split(' '))<4 :
            print(line)
            continue
        testList.append(line.strip())
    vectors = load_vectors()
    return testList, vectors

def load_train_and_vectors():
    trainList = []
    for line in open(train_data,encoding='utf-8'):
        trainList.append(line.strip())
    vectors = load_vectors()
    return trainList, vectors

def load_data_val_10(testList, vectors, index):
    x_train_1 = []
    x_train_2 = []
    x_train_3 = []
    items = testList[index].split(' ')
    x_train_1.append(vocab_plus_overlap(vectors, items[2], items[3], 200))
    x_train_2.append(vocab_plus_overlap(vectors, items[3], items[2], 200))
    x_train_3.append(vocab_plus_overlap(vectors, items[3], items[2], 200))
    return np.array(x_train_1), np.array(x_train_2), np.array(x_train_3)

# 读按词分割的正例
def read_raw():
    raw = []
    for line in open(train_data,encoding='utf-8'):
        items = line.strip().split(' ')
        if items[0] == '1':
            raw.append(items)
    return raw

# 读取词对应的onehot的编号
def encode_sent(vocab, string, size):
    x = []
    words = string.split('_')
    for i in range(0, size):
        if len(words)>size and words[i] in vocab :
            x.append(vocab[words[i]])
        else:
            x.append(vocab['UNKNOWN'])
    return x


# 主要读取函数
# 每次随机拿200条数据来训练。。。
# 分别是 问题，正例，负例
# 负例是随机拿了一个答案。。。emmmm
def load_data_6(vocab, alist, raw, size, seq_size=200):
    x_train_1 = []
    x_train_2 = []
    x_train_3 = []
    for i in range(0, size):
        items = raw[random.randint(0, len(raw) - 1)]
        nega = rand_qa(alist)
        x_train_1.append(encode_sent(vocab, items[2], seq_size))
        x_train_2.append(encode_sent(vocab, items[3], seq_size))
        x_train_3.append(encode_sent(vocab, nega, seq_size))
    return np.array(x_train_1), np.array(x_train_2), np.array(x_train_3)

def load_data_val_6(testList, vocab, index, batch, seq_size=200):
    x_train_1 = []
    x_train_2 = []
    x_train_3 = []
    for i in range(0, batch):
        true_index = index + i
        if (true_index >= len(testList)):
            true_index = len(testList) - 1
        items = testList[true_index].split(' ')
        # if len(items)<4:continue
        x_train_1.append(encode_sent(vocab, items[2], seq_size))
        x_train_2.append(encode_sent(vocab, items[3], seq_size))
        x_train_3.append(encode_sent(vocab, items[3], seq_size))
    return np.array(x_train_1), np.array(x_train_2), np.array(x_train_3)

def load_data_9(trainList, vectors, size):
    x_train_1 = []
    x_train_2 = []
    y_train = []
    for i in range(0, size):
        pos = trainList[random.randint(0, len(trainList) - 1)]
        posItems = pos.strip().split(' ')
        x_train_1.append(vocab_plus_overlap(vectors, posItems[2], posItems[3], 200))
        x_train_2.append(vocab_plus_overlap(vectors, posItems[3], posItems[2], 200))
        y_train.append([1, 0])
        neg = trainList[random.randint(0, len(trainList) - 1)]
        negItems = neg.strip().split(' ')
        x_train_1.append(vocab_plus_overlap(vectors, posItems[2], negItems[3], 200))
        x_train_2.append(vocab_plus_overlap(vectors, negItems[3], posItems[2], 200))
        y_train.append([0, 1])
    return np.array(x_train_1), np.array(x_train_2), np.array(y_train)

def load_data_val_9(testList, vectors, index):
    x_train_1 = []
    x_train_2 = []
    items = testList[index].split(' ')
    x_train_1.append(vocab_plus_overlap(vectors, items[2], items[3], 200))
    x_train_2.append(vocab_plus_overlap(vectors, items[3], items[2], 200))
    return np.array(x_train_1), np.array(x_train_2)

def load_data_10(vectors, qalist, raw, size):
    x_train_1 = []
    x_train_2 = []
    x_train_3 = []
    items = raw[random.randint(0, len(raw) - 1)]
    nega = rand_qa(qalist)
    x_train_1.append(vocab_plus_overlap(vectors, items[2], items[3], 200))
    x_train_2.append(vocab_plus_overlap(vectors, items[3], items[2], 200))
    x_train_3.append(vocab_plus_overlap(vectors, nega, items[2], 200))
    return np.array(x_train_1), np.array(x_train_2), np.array(x_train_3)

def load_data_11(vectors, qalist, raw, size):
    x_train_1 = []
    x_train_2 = []
    x_train_3 = []
    items = raw[random.randint(0, len(raw) - 1)]
    nega = rand_qa(qalist)
    x_train_1.append(vocab_plus_overlap(vectors, items[2], items[3], 200))
    x_train_2.append(vocab_plus_overlap(vectors, items[3], items[2], 200))
    x_train_3.append(vocab_plus_overlap(vectors, nega, items[2], 200))
    return np.array(x_train_1), np.array(x_train_2), np.array(x_train_3)

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
    

def deal_Chinese():
    rawlist=open("D:/新建文件夹/泰迪杯数据挖掘竞赛/data/C题全部数据/train_data_complete.json", encoding="utf-8", mode="r").readlines()
    data=json.loads("".join(rawlist))
    of=open("D:/新建文件夹/泰迪杯数据挖掘竞赛/data/C题部分数据/test_data_50",mode="w",encoding="utf-8")
   #  of2=open("D:/新建文件夹/泰迪杯数据挖掘竞赛/data/C题部分数据/test_data_allright",mode="w",encoding="utf-8")
    print(data[0])
    res=list()
    res_right=list()
    item_num=0
    for item in data:
        if item_num > 50: break
        item_num+=1
        q=list(jieba.cut(item['question'].replace(' ','').replace('_','')))
        qid=item['item_id']
        for i in item['passages']:

            a=list(jieba.cut(i['content'].replace(' ','').replace('_','')))
            label=i['label']
            if label == 1:
                res.append("%s qid:%s %s %s\n" % (1,qid,'_'.join(q),'_'.join(a)))
                # print("%s qid:%s %s %s" % (1,qid,'_'.join(q),'_'.join(a)))
                res_right.append("%s qid:%s %s %s\n" % (1, qid, '_'.join(q), '_'.join(a)))
            else:
                res.append("%s qid:%s %s %s\n" % (0, qid, '_'.join(q), '_'.join(a)))
    of.writelines(res)
    # of2.writelines(res_right)
    of.close()
    # of2.close()
    print("over")
    # print(data[0])
    # code=int(0)
    # vocab = {}
    # vocab['UNKNOWN'] = code
    # code += 1
    # for line in open('insuranceQA/train'):
    #     cut=jieba.cut(line)


# deal_Chinese()