# -*- encoding:utf-8 -*-
import random
from qacnn import QACNN
import tensorflow as tf
import numpy as np
import datetime
import my_qa as qa

# Config函数
class Config(object):
    def __init__(self, vocab_size):
        # 输入序列(句子)长度
        self.sequence_length = 50
        # 循环数
        # self.num_epochs = 100000
        self.num_epochs=100000
        # batch大小
        self.batch_size =200
        # 词表大小
        self.vocab_size = vocab_size
        # 词向量大小
        self.embedding_size = 80
        # 不同类型的filter,相当于1-gram,2-gram,3-gram和5-gram
        self.filter_sizes = [1, 2, 3, 5]
        # 隐层大小
        self.hidden_size = 80
        # 每种filter的数量
        self.num_filters = 512
        # L2正则化  论文里给的是0.0001
        self.l2_reg_lambda = 0.0001
        # 弃权
        self.keep_prob = 0.75
        # 学习率
        # 论文里给的是0.01
        self.lr = 0.01
        # margin
        # 论文里给的是0.009
        self.m = 0.05
        # 设置log打印
        self.cf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        # 占用GPU内存比例
        self.cf.gpu_options.per_process_gpu_memory_fraction = 0.7

        self.global_step = tf.Variable(0, name="global_step", trainable=False)



global saver,sess,config,wvmodel,tfidf,train_data,test_data,all_answers
def train_init():
    print( 'Loading Data...')
    global wvmodel, tfidf, train_data, test_data, all_answers,saver,config,sess
    wvmodel=qa.load_wvmodel()
    #  tfidf=qa.load_tfidf()
    train_data=qa.load_data(qa.chinese_json_train)
    test_data=qa.load_data(qa.chinese_json_train2,500)
    all_answers=qa.load_all_answers(train_data)
    print( 'Loading Data Done!')
    # 配置文件
    config = Config(wvmodel.vector_size)
    config.cf.gpu_options.allow_growth = True
    sess=tf.Session(config=config.cf)



def train(init=True):
    train_init()
    global sess
    # 开始训练和测试
    with tf.device('/gpu:0'):
        # 建立CNN网络
        cnn = QACNN(config, sess)
        # 训练函数
        def train_step(x_batch_1, x_batch_2, x_batch_3):
            feed_dict = {
                cnn.q: x_batch_1,
                cnn.aplus: x_batch_2,
                cnn.aminus: x_batch_3,
                cnn.keep_prob: config.keep_prob
            }
            _, step, loss, accuracy = sess.run(
                [cnn.train_op, cnn.global_step, cnn.loss, cnn.accu],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print( "{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            return step,loss,accuracy
        # 测试函数
        def dev_step(th=0.75):
            # global sess
            i = 0
            tt_num = .0
            tf_num = .0
            ft_num = .0
            ff_num = .0
            while True:
                x_test_1, x_test_2, y_label = qa.get_test_x(test_data, wvmodel, all_answers, config.batch_size, i,
                                                            config.sequence_length)
                feed_dict = {
                    cnn.q: x_test_1,
                    cnn.aplus: x_test_2,
                    cnn.aminus: x_test_2,  # 这一数据占位，并没实际计算
                    cnn.keep_prob: 1.0
                }
                batch_scores = sess.run([cnn.q_ap_cosine], feed_dict)
                for j in range(0, len(batch_scores[0])):
                    s = batch_scores[0][j]
                    l = y_label[j]
                    # print("%s-%s,预测%s,实际%s"%(i,j,s,l))
                    if s >= th and l == 1:
                        tt_num += 1
                    elif s >= th and l == 0:
                        tf_num += 1
                    elif s < th and l == 1:
                        ft_num += 1
                    elif s < th and l == 0:
                        ff_num += 1
                i += 1
                if i >= min(10,len(test_data)):
                    break
            print("tt %s,tf %s,ft %s,ff %s" % (tt_num, tf_num, ft_num, ff_num))
            acc = tt_num / (tt_num + tf_num)
            cbk = tt_num / (tt_num + ft_num)
            print('准确率%s,召回率%s,F1值%s ' % (acc, cbk, (2 * acc * cbk) / (acc + cbk)))
            print('F1=%s' % (tt_num * 2 / (tt_num + tf_num + ft_num)))
        # 测试间隔
        evaluate_every = 500000
        save_every=50
        # 开始训练
        if init==True:
            sess.run(tf.global_variables_initializer())
        else:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint('data/'))

        saver = tf.train.Saver()
        for i in range(config.num_epochs):
            # 获取batch
            x_batch_1, x_batch_2, x_batch_3 = qa.get_random_train_x(train_data,all_answers,wvmodel,config.batch_size,config.sequence_length)

            # 训练
            gstep,gloss,gaccuracy= train_step(x_batch_1, x_batch_2, x_batch_3)

            # 保存
            if (i+1) % save_every == 0:
                saver.save(sess, 'data/qa_model',global_step=gstep)
                print("Saved model checkpoint %s,acc:%s" % (gstep, gaccuracy))

            # 测试
            if (i+1) % evaluate_every == 0:
                print("\n测试{}:".format((i+1)/evaluate_every))
                dev_step()
                print()

def load(th=0.5,path='data/'):
    global sess
    with tf.device('/gpu:0'):
        test_data = qa.load_data(qa.chinese_json_train2,500)
        wvmodel = qa.load_wvmodel()
        config = Config(wvmodel.vector_size)
        config.cf.gpu_options.allow_growth = True
        sess = tf.Session(config=config.cf)
        cnn = QACNN(config, sess)
        all_answers=qa.load_all_answers(test_data)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(path))
        print("ModelV restored.")
        batch_size=200
        def dev_step():
            i = 0
            tt_num=.0
            tf_num=.0
            ft_num=.0
            ff_num=.0
            while True:
                x_test_1, x_test_2, y_label = qa.get_test_x(test_data, wvmodel, all_answers,batch_size, i, config.sequence_length)
                feed_dict = {
                    cnn.q: x_test_1,
                    cnn.aplus: x_test_2,
                    cnn.aminus: x_test_2,  # 这一数据占位，并没实际计算
                    cnn.keep_prob: 1.0
                }
                batch_scores = sess.run([cnn.q_ap_cosine], feed_dict)

                for j in range(0,min(len(batch_scores[0]),100)):
                    s=batch_scores[0][j]
                    l=y_label[j]
                    # print("%s-%s,预测%s,实际%s"%(i,j,s,l))
                    if s >= th and l == 1:
                        tt_num+=1
                    elif s >= th and l == 0:
                        tf_num+=1
                    elif s<th and l == 1:
                        ft_num+=1
                    elif s<th and l == 0:
                        ff_num+=1
                i += 1 # config.batch_size
                acc = tt_num / (tt_num + tf_num)
                cbk = tt_num / (tt_num + ft_num)
                # print('%s 准确率%s,召回率%s,F1值%s' % (i, acc, cbk, tt_num * 2 / (tt_num + tf_num + ft_num)))
                if i >= len(test_data):
                    break
            # 回答的正确数和错误数
            print("tt %s,tf %s,ft %s,ff %s" % (tt_num,tf_num,ft_num,ff_num))
            acc=tt_num/(tt_num+tf_num)
            cbk=tt_num/(tt_num+ft_num)
            print('准确率%s,召回率%s,F1值%s' % (acc,cbk,tt_num * 2 /(tt_num+tf_num+ft_num)))
            # print('F1=%s' % (tt_num * 2 /(tt_num+tf_num+ft_num)))
        # 测试
        print("\n测试:")

        dev_step()




def test_baseline1(th=0.5):
    wvmodel = qa.load_wvmodel()
    tfidf=qa.load_tfidf()
    tt_num = .0
    tf_num = .0
    ft_num = .0
    ff_num = .0
    batch_size = 100
    test_data=qa.load_data(qa.chinese_json_train2,50)
    all_answers = qa.load_all_answers(test_data)
    for i in range(0,len(test_data)):
        q=test_data[i][0]
        a_list=test_data[i][1]
        n_list=test_data[i][2]
        for j in range(0,batch_size):
            if j < len(a_list):
                a=a_list[j]
                l=1
            elif j < len(a_list)+len(n_list):
                a=n_list[j-len(a_list)]
                l=0
            else:
                a=all_answers[random.randint(0,len(all_answers)-1)]
                l=0
            q_v = qa.get_sentence_vector_baseline(wvmodel, tfidf, q)
            a_v = qa.get_sentence_vector_baseline(wvmodel, tfidf, a)
            q_v=np.array(q_v)
            a_v=np.array(a_v)
            s = np.dot(q_v, a_v) / (np.linalg.norm(q_v) * np.linalg.norm(a_v))
            # print("%s-%s,预测%s,实际%s" % (i, j, s, l))
            if s >= th and l == 1:
                tt_num += 1
            elif s >= th and l == 0:
                tf_num += 1
            elif s < th and l == 1:
                ft_num += 1
            elif s < th and l == 0:
                ff_num += 1
        acc = tt_num / (tt_num + tf_num)
        cbk = tt_num / (tt_num + ft_num)
        # print('%s 准确率%s,召回率%s,F1值%s' % (i, acc, cbk, tt_num * 2 / (tt_num + tf_num + ft_num)))
    print("tt %s,tf %s,ft %s,ff %s" % (tt_num, tf_num, ft_num, ff_num))
    acc = tt_num / (tt_num + tf_num)
    cbk = tt_num / (tt_num + ft_num)
    print('准确率%s,召回率%s,F1值%s ' % (acc, cbk, (2 * acc * cbk) / (acc + cbk)))
    print('F1=%s' % (tt_num * 2 / (tt_num + tf_num + ft_num)))


def test_output_baseline(th=0.5):
    wvmodel = qa.load_wvmodel()
    tfidf = qa.load_tfidf()
    test_data = qa.load_data("D:/新建文件夹/泰迪杯数据挖掘竞赛/data/C题测试数据/test_data_complete.json")

    output=open('data/test_result',mode='w',encoding='utf-8')

    for item in test_data:
        q=item[0]
        # qid=item[4]
        a_list=item[1]
        a_ids=item[3]
        for i in range(0,len(a_list)):
            a = a_list[i]
            aid = a_ids[i]
            q_v = qa.get_sentence_vector_baseline(wvmodel, tfidf, q)
            a_v = qa.get_sentence_vector_baseline(wvmodel, tfidf, a)
            q_v = np.array(q_v)
            a_v = np.array(a_v)
            s = np.dot(q_v, a_v) / (np.linalg.norm(q_v) * np.linalg.norm(a_v))
            if s >= th:
                res = 1
            else:
                res = 0
            output.write("%s,%s\n" % (aid,res))

    output.close()
    # all_answers = qa.load_all_answers(test_data)



if __name__ == '__main__':
    # load()
    # load('old-model/')
    # load('old-model-2/')
    # train(init=False)
    test_output_baseline(0.5)

    # th=0.4
    # while True:
    #     print(th)
    #     # test_baseline1(th)
    #     load(th)
    #     th+=0.05
    #     if th > 0.6:break