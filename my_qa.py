from __future__ import print_function
import numpy as np
import random
import jieba
import json
from gensim import corpora,similarities,models

train_data="insuranceQA/train_data_allright_small"
test_data="insuranceQA/test_data_50"
# vector_data="insuranceQA/vectors.nobin"
# vector_data="D:/新建文件夹/word2vec/news12g_bdbk20g_nov90g_dim128/news_12g_baidubaike_20g_novel_90g_embedding_64.model"
vector_data="D:/新建文件夹/word2vec/w2c-python-gensim版/wiki.zh.text.model"
stop_words="insuranceQA/stopwords"

chinese_json_train="D:/新建文件夹/泰迪杯数据挖掘竞赛/data/C题全部数据/train_data_complete.json"
chinese_json_train2="D:/新建文件夹/泰迪杯数据挖掘竞赛/data/C题部分数据/train_data_sample.json"

def get_word_list():
    rawlist = open(chinese_json_train2, encoding="utf-8", mode="r").readlines()
    data = json.loads("".join(rawlist))
    # print(data[0])
    res = list()
    res_right = list()
    res_allsen=list()
    item_num = 0
    for item in data:
        if item_num > 500: break
        item_num += 1
        q = list(jieba.cut(item['question'].replace(' ', '').replace('_', '')))
        if len(q) > 1: res_allsen.append(q)
        qid = item['item_id']
        for i in item['passages']:
            a = list(jieba.cut(i['content'].replace(' ', '').replace('_', '')))
            if len(a)>1: res_allsen.append(a)
            # label = i['label']
            # if label == 1:
            #     res.append("%s qid:%s %s %s\n" % (1, qid, '_'.join(q), '_'.join(a)))
            #     res_right.append("%s qid:%s %s %s\n" % (1, qid, '_'.join(q), '_'.join(a)))
            # else:
            #     res.append("%s qid:%s %s %s\n" % (0, qid, '_'.join(q), '_'.join(a)))
    print("load over")
    return res_allsen

def get_stop_words():
    stop_dict=dict()
    for line in open(stop_words,encoding="utf-8").readlines():
        line=line.strip()
        if len(line) > 0:
            stop_dict[line]=1
    return stop_dict

def get_sentence_vector(wvmodel,tfidf,sentence):
    sentence=list(jieba.cut(sentence))
    # print(tfidf)
    sv=list()
    for i in range(0,wvmodel.vector_size): sv.append(0.0)
    for word in sentence:
        # print(word in w2v.wv,word in tfidf)
        if word in wvmodel.wv and word in tfidf:
            tmp=list()
            for v in wvmodel.wv[word]:
                tmp.append(v*tfidf[word])
            # print(tmp)
            for i in range(0,len(sv)):
                sv[i]=sv[i]+tmp[i]
            # sv.append(tmp)
    # print(sv)
    # svr=list()
    return sv


def load_wvmodel():
    w2v = models.Word2Vec.load(vector_data)
    print("w2v model load over")
    return w2v

def load_tfidf():
    corpora_documents = []
    corpora_documents_all = []
    # 停用词
    stop_dict = get_stop_words()
    # 分词处理
    for item_text in get_word_list():
        # item_seg = list(jieba.cut(item_text))
        res_list = list()
        corpora_documents_all.append(item_text)
        for word in item_text:
            if word not in stop_dict:
                res_list.append(word)
        corpora_documents.append(res_list)



        # 生成字典和向量语料
    print("corpora count:%s" % len(corpora_documents))
    dictionary = corpora.Dictionary(corpora_documents)
    print("dict count:%s" % len(dictionary))
    # print(dictionary)
    # dictionary.save('dict.txt') #保存生成的词典
    # dictionary=Dictionary.load('dict.txt')#加载

    # 通过下面一句得到语料中每一篇文档对应的稀疏向量（这里是bow向量）
    corpus = [dictionary.doc2bow(text) for text in corpora_documents]
    # 向量的每一个元素代表了一个word在这篇文档中出现的次数
    # print(corpus[0])

    # corpora.MmCorpus.serialize('corpuse.mm',corpus)#保存生成的语料
    # corpus=corpora.MmCorpus('corpuse.mm')#加载

    # corpus是一个返回bow向量的迭代器。下面代码将完成对corpus中出现的每一个特征的IDF值的统计工作
    tfidf_model = models.TfidfModel(corpus)
    corpus_tfidf = tfidf_model[corpus]

    # 输出排序的tfidf结果
    tmpd = dict()
    for item in corpus_tfidf:
        for word in item:
            tmpd[word[0]] = word[1]
    tmpl = list()
    for k in tmpd:
        tmpl.append([dictionary[k], tmpd[k]])
    tfidf_res = sorted(tmpl, key=lambda item: item[1], reverse=True)
    print(tfidf_res[0:10])
    tfidf = dict()
    for item in tfidf_res:
        tfidf[item[0]] = item[1]

    print ("tfidf load over")
    return tfidf


def qa_1():
    corpora_documents = []
    corpora_documents_all=[]
    # 停用词
    stop_dict=get_stop_words()
    # 分词处理
    for item_text in get_word_list():
        #item_seg = list(jieba.cut(item_text))
        res_list=list()
        corpora_documents_all.append(item_text)
        for word in item_text:
            if word not in stop_dict:
                res_list.append(word)
        corpora_documents.append(res_list)



        # 生成字典和向量语料
    print("corpora count:%s" % len(corpora_documents))
    dictionary = corpora.Dictionary(corpora_documents)
    print("dict count:%s" % len(dictionary))
    print(dictionary)
    # dictionary.save('dict.txt') #保存生成的词典
    # dictionary=Dictionary.load('dict.txt')#加载

    # 通过下面一句得到语料中每一篇文档对应的稀疏向量（这里是bow向量）
    corpus = [dictionary.doc2bow(text) for text in corpora_documents]
    # 向量的每一个元素代表了一个word在这篇文档中出现的次数
    print(corpus[0])

    # corpora.MmCorpus.serialize('corpuse.mm',corpus)#保存生成的语料
    # corpus=corpora.MmCorpus('corpuse.mm')#加载

    # corpus是一个返回bow向量的迭代器。下面代码将完成对corpus中出现的每一个特征的IDF值的统计工作
    tfidf_model = models.TfidfModel(corpus)
    corpus_tfidf = tfidf_model[corpus]

    # 输出排序的tfidf结果
    tmpd=dict()
    for item in corpus_tfidf:
        for word in item:
            tmpd[word[0]]=word[1]
    tmpl=list()
    for k in tmpd:
        tmpl.append([dictionary[k],tmpd[k]])
    tfidf_res=sorted(tmpl,key=lambda item:item[1],reverse=True)
    print(tfidf_res[0:10])
    tfidf=dict()
    for item in tfidf_res:
        tfidf[item[0]]=item[1]


    #查看model中的内容 
    # for item in corpus_tfidf:
    #      print(item)
    # tfidf.save("data.tfidf") 
    # tfidf = models.TfidfModel.load("data.tfidf") 
    #print(tfidf_model.dfs)

    #return

    sents=dict()
    w2v=models.Word2Vec.load(vector_data)
    # print(w2v)
    # w2v.build_vocab(sents, update=True)
    # print(sents)
    # print(w2v)
    # print(w2v.vector_size)


    sentence = "高速占用应急车道行驶扣几分"
    res = get_sentence_vector(w2v, tfidf, sentence)
    print(res)

    similarity = similarities.Similarity('Similarity-tfidf-index', corpus_tfidf, num_features=len(dictionary))
    test_data_1 = '高速公路的应急车道是“生命通道”'
    test_cut_raw_1 = list(jieba.cut(test_data_1))  # ['北京', '雾', '霾', '红色', '预警']
    test_corpus_1 = dictionary.doc2bow(test_cut_raw_1)  # [(51, 1), (59, 1)]，即在字典的56和60的地方出现重复的字段，这个值可能会变化
    similarity.num_best = 5
    test_corpus_tfidf_1 = tfidf_model[test_corpus_1]  # 根据之前训练生成的model，生成query的IFIDF值，然后进行相似度计算
    # [(51, 0.7071067811865475), (59, 0.7071067811865475)]
    print(similarity[test_corpus_tfidf_1])  # 返回最相似的样本材料,(index_of_document, similarity) tuples
    sindex=similarity[test_corpus_tfidf_1][0][0]
    print(sindex)
    print(''.join(corpora_documents_all[sindex]))
    print('-' * 40)

    # 使用LSI模型进行相似度计算
    lsi = models.LsiModel(corpus_tfidf)
    corpus_lsi = lsi[corpus_tfidf]
    similarity_lsi = similarities.Similarity('Similarity-LSI-index', corpus_lsi, num_features=400, num_best=2)
    # save
    # LsiModel.load(fname, mmap='r')#加载
    test_data_3 = '高速公路的应急车道是“生命通道”'
    test_cut_raw_3 = list(jieba.cut(test_data_3))  # 1.分词
    test_corpus_3 = dictionary.doc2bow(test_cut_raw_3)  # 2.转换成bow向量
    test_corpus_tfidf_3 = tfidf_model[test_corpus_3]  # 3.计算tfidf值
    test_corpus_lsi_3 = lsi[test_corpus_tfidf_3]  # 4.计算lsi值
    # lsi.add_documents(test_corpus_lsi_3) #更新LSI的值
    print('-' * 40)
    print(similarity[test_corpus_lsi_3])
    print(''.join(corpora_documents_all[similarity[test_corpus_lsi_3][0][0]]))


qa_1()