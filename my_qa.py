from __future__ import print_function
import random
import jieba
import json
from gensim import corpora,similarities,models

# vector_data="D:/新建文件夹/word2vec/word2vec_from_weixin/word2vec/word2vec_wx"
vector_data="D:/新建文件夹/word2vec/w2c-python-gensim版/wiki.zh.text.model"
stop_words="data/stopwords"

chinese_json_train="D:/新建文件夹/泰迪杯数据挖掘竞赛/data/C题全部数据/train_data_complete.json"
chinese_json_train2="D:/新建文件夹/泰迪杯数据挖掘竞赛/data/C题部分数据/train_data_sample.json"

tfidf_data="data/tfidf"

def get_word_list(size=-1):
    rawlist = open(chinese_json_train, encoding="utf-8", mode="r").readlines()
    data = json.loads("".join(rawlist))
    # print(data[0])
    res = list()
    res_right = list()
    res_allsen=list()
    item_num = 0
    for item in data:
        if size>0 and item_num > size: break
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
    print("load json over")
    return res_allsen

def get_stop_words():
    stop_dict=dict()
    for line in open(stop_words,encoding="utf-8").readlines():
        line=line.strip()
        if len(line) > 0:
            stop_dict[line]=1
    return stop_dict

def get_sentence_vector_baseline(wvmodel,tfidf,sentence):
    # if len(sentence):
    # sentence=list(jieba.cut(sentence))
    # print(tfidf)
    sv = list()
    for i in range(0,wvmodel.vector_size): sv.append(0.0)
    for word in sentence:
        # print(word in w2v.wv,word in tfidf)
        if word in wvmodel.wv and str(word) in tfidf:
            tmp=list()
            for v in wvmodel.wv[word]:
                d=v * tfidf[word]
                # print(d)
                tmp.append(d)
            # print(tmp)
            for i in range(0,len(sv)):
                sv[i] = sv[i] + tmp[i]
            # sv.append(tmp)
    # print(sv)
    # svr=list()
    return sv

def get_sentence_vectors(wvmodel,sentence,sentence_num):
    sv = list()
    # for i in range(0,wvmodel.vector_size): sv.append(0.0)
    i=0
    for word in sentence:
        if i >= sentence_num: break
        # print(word in w2v.wv,word in tfidf)
        if word in wvmodel.wv:
            i += 1
            sv.append(wvmodel.wv[word])
    for i in range(len(sv),sentence_num):
        t=list()
        for j in range(0,wvmodel.vector_size):
            t.append(0.0)
        sv.append(t)
    # print(len(sv),len(sv[0]),len(sv[len(sv)-1]))
    # svr=list()
    return sv

def load_data(fname,size=-1):
    rawlist = open(fname, encoding="utf-8", mode="r").readlines()
    data = json.loads("".join(rawlist))
    res = list()
    item_num = 0
    for item in data:

        if size > 0 and item_num > size: break
        item_num += 1

        qa_item=list()

        q = list(jieba.cut(item['question'].replace(' ', '').replace('_', '')))
        if len(q) <1:continue
        # qa_item.append(q)
        qa_a=list()
        qa_n=list()
        ids=list()
        # id_q=item['item_id']
        for i in item['passages']:
            a = list(jieba.cut(i['content'].replace(' ', '').replace('_', '')))
            # if len(a) > 1: res_allsen.append(a)
            id=''
            if 'passage_id' in i:
                id=i['passage_id']
            ids.append(id)
            if 'label' in i:
                label = i['label']
                if label == 1:
                    # +
                    qa_a.append(a)
                    # res.append("%s qid:%s %s %s\n" % (1, qid, '_'.join(q), '_'.join(a)))
                    # res_right.append("%s qid:%s %s %s\n" % (1, qid, '_'.join(q), '_'.join(a)))
                else:
                    # -
                    qa_n.append(a)
                    # res.append("%s qid:%s %s %s\n" % (0, qid, '_'.join(q), '_'.join(a)))
            else:
                qa_a.append(a)
        if len(qa_a)<=0:continue
        qa_item.append(q)
        qa_item.append(qa_a)
        qa_item.append(qa_n)
        qa_item.append(ids)
        # qa_item.append(id_q)
        res.append(qa_item)

    print("load data %s over" % fname)
    return res

def load_all_answers(data,size=-1):
    # rawlist = open(chinese_json_train2, encoding="utf-8", mode="r").readlines()
    # data = json.loads("".join(rawlist))
    res = list()
    item_num = 0
    for item in data:
        for a in item[1]:
            res.append(a)
        for n in item[2]:
            res.append(n)
        # if size>0 and item_num > size: break
        # item_num += 1
        # for i in item['passages']:
        #     a = list(jieba.cut(i['content'].replace(' ', '').replace('_', '')))
        #     label = i['label']
        #     res.append(a)
    print("load all answers over")
    return res

def load_wvmodel():
    w2v = models.Word2Vec.load(vector_data)
    print("w2v model load over")
    return w2v

def create_tfidf():
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

    # 得到语料中每一篇文档对应的bow向量
    corpus = [dictionary.doc2bow(text) for text in corpora_documents]
    # 向量的每一个元素代表了一个word在这篇文档中出现的次数
    # print(corpus[0])
    # corpora.MmCorpus.serialize('corpuse.mm',corpus)#保存生成的语料
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

    # save
    save_file=open(tfidf_data,mode="w",encoding="utf-8")
    for line in tfidf_res:
       save_file.write("%s\t%s\n" % (line[0],line[1]))
    save_file.close()


    print ("tfidf create over")

    # return tfidf


def load_tfidf():
    f=open(tfidf_data,mode='r',encoding='utf-8')
    tfidf_lines=f.readlines()
    tfidf = dict()
    for item in tfidf_lines:
        tmp_line=item.strip().split('\t')
        if len(tmp_line)<2:continue
        tfidf[str(tmp_line[0])] = float(tmp_line[1])
    print("tfidf load over")
    return tfidf



def get_random_train_x(train_data,all_answers,wvmodel,size,sentence_size):
    # 选取反例时候，有多大比例从原问题对应的反例里选
    n_p=0.75
    x_train_1=list()
    x_train_2=list()
    x_train_3=list()
    for i in range(0,size):
        # print(i)
        index = random.randint(0, len(train_data) - 1)
        q=train_data[index][0]
        qa_a=train_data[index][1]
        qa_n=train_data[index][2]

        a_index=random.randint(0,len(qa_a)-1)
        # a_index = 0

        if len(qa_n)<1 or random.random()<n_p:
            # random a_n
            n_index = random.randint(0, len(all_answers) - 1)
            n = all_answers[n_index]
        else:
            # from self qa_n list
            n = qa_n[random.randint(0,len(qa_n)-1)]
        a=qa_a[a_index]


        # x_train_1_line = get_sentence_vector(wvmodel, tfidf, q)
        # x_train_2_line = get_sentence_vector(wvmodel, tfidf, a)
        # x_train_3_line = get_sentence_vector(wvmodel, tfidf, n)

        x_train_1_line = get_sentence_vectors(wvmodel, q,sentence_size)
        x_train_2_line = get_sentence_vectors(wvmodel, a,sentence_size)
        x_train_3_line = get_sentence_vectors(wvmodel, n,sentence_size)

        x_train_1.append(x_train_1_line)
        x_train_2.append(x_train_2_line)
        x_train_3.append(x_train_3_line)
    # print(x_train_1)
    return  x_train_1,x_train_2,x_train_3

def get_test_x(test_data,wvmodel,all_answers,size,index,sentence_size):
    x_train_1 = list()
    x_train_2 = list()
    y_label=list()
    # x_train_3 = list()

    test_data_line=test_data[index]
    q=test_data_line[0]
    qa_a=test_data_line[1]
    qa_n=test_data_line[2]

    for i in range(0,size):
        a = list()
        label=0
        if i < len(qa_a):
            a = qa_a[i]
            label=1
        elif i-len(qa_a) < len(qa_n):
            a = qa_n[i-len(qa_a)]
            label=0
        else:
            a = all_answers[random.randint(0, len(all_answers)-1)]
            label=0
        x_train_1.append(get_sentence_vectors(wvmodel,q,sentence_size))
        x_train_2.append(get_sentence_vectors(wvmodel,a,sentence_size))
        y_label.append(label)
    return x_train_1, x_train_2, y_label

#
# # w2v+tfidf权重 作对比
# def test_baseline1(test_data,wvmodel,tfidf,all_answers,size,index,sentence_size):
#     x_train_1 = list()
#     x_train_2 = list()
#     y_label=list()
#
#     test_data_line=test_data[index]
#     q=test_data_line[0]
#     qa_a=test_data_line[1]
#     qa_n=test_data_line[2]
#
#     for i in range(0,size):
#         a = list()
#         label=0
#         if i < len(qa_a):
#             a = qa_a[i]
#             label=1
#         elif i-len(qa_a) < len(qa_n):
#             a = qa_n[i-len(qa_a)]
#             label=0
#         else:
#             a = all_answers[random.randint(0, len(all_answers)-1)]
#             label=0
#         x_train_1.append(get_sentence_vector_baseline(wvmodel,q,sentence_size))
#         x_train_2.append(get_sentence_vector_baseline(wvmodel,a,sentence_size))
#         y_label.append(label)
#     return x_train_1, x_train_2, y_label

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


    # sentence = "高速占用应急车道行驶扣几分"
    # res = get_sentence_vector(w2v, tfidf, list(jieba.cut(sentence)))
    #print(res)

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


# qa_1()


if __name__ == '__main__':
    load_tfidf()
    # create_tfidf()