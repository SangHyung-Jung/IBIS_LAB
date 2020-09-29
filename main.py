'''
이 py는 이제 LSTM_T_CNN_C임
우리 모델 학습하고 test하는 부분

back_testing이랑 합쳐도 될꺼같긴해

우리 data2.pikle이라는 파일이 있었는데
날아가서 내가 train data 마지막에 쓴거 zip 파일로 보냄

'''

import tensorflow as tf
from gensim.models.word2vec import Word2Vec
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import os, pickle, time, sys
import random
import torch
import csv

# with open('/home/ir1067/Result/data2.pickle', 'rb') as f:
#     data_new_vali2 = pickle.load(f)
"""
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0
"""

# data_new_vali2['train']['CJ대한통운']['indicator'][353][13]
# data_new_vali2 = get_xy_7([('train', '2014-01-01', '2016-12-31'),
#                         ('vali', '2017-01-01', '2017-12-31'),
#                         ('test', '2018-01-01', '2018-12-31')])

# with open('/home/ir1067/Result/data2.pickle', 'wb') as f:
#     pickle.dump(data_new_vali2, f)

with open('/home/ir1067/Result/data2.pickle', 'rb') as f:
    data_new_vali2 = pickle.load(f)

with open(r'data.pickle', 'rb') as f:
    data_new_vali2 = pickle.load(f)
# with open('/home/ir1067/Result/data10.pickle', 'rb') as f:
#     data_new_vali2 = pickle.load(f)
# with open('/home/ir1067/Result/data30.pickle', 'rb') as f:
#     data_new_vali2 = pickle.load(f)

def TimeEffect(dic, w2v, batch_size, embed_size, input_size, input_size2, companies=None, num=0):
    X_titles = []
    X_contents = []
    y = []
    X_indi = []
    company_list = list(dic.keys())[num:num + 1]

    if companies == None:
        for company in company_list:
            X_titles += dic[company]['titles']
            X_contents += dic[company]['contents']
            y += dic[company]['label']
            X_indi += dic[company]['indicator']
    elif type(companies) == list:
        for company in companies:
            X_titles += dic[company]['titles']
            X_contents += dic[company]['contents']
            y += dic[company]['label']
            X_indi += dic[company]['indicator']
    else:
        for company in companies:
            X_titles += dic[company]['titles']
            X_contents += dic[company]['contents']
            y += dic[companies]['label']
            X_indi += dic[company]['indicator']

    # shuffle
    random.seed(42)
    random.shuffle(X_titles)
    random.seed(42)
    random.shuffle(X_contents)
    random.seed(42)
    random.shuffle(y)
    random.seed(42)
    random.shuffle(X_indi)

    length = len(X_titles)
    max_batch = length // batch_size + 1
    index = 0
    i = 0

    while index < length:
        batch_x_titles = []
        batch_x_contents = []
        try:
            batch_y = y[index: index + batch_size]
            temp_x_titles = X_titles[index: index + batch_size]
            temp_x_contents = X_contents[index: index + batch_size]
            batch_indi = X_indi[index: index + batch_size]
        except IndexError:
            batch_y = y[index:]
            temp_x_titles = X_titles[index:]
            temp_x_contents = X_contents[index:]
            batch_indi = X_indi[index: index + batch_size]

        for news in temp_x_titles:
            arr = np.zeros((input_size2, embed_size))
            idx = 0

            for word in news:
                try:
                    vec = w2v.wv[word]
                    arr[idx] = vec
                    idx += 1
                    del vec
                    if idx == input_size2:
                        break
                except KeyError:
                    pass
            batch_x_titles.append(arr)

        for news in temp_x_contents:
            arr = np.zeros((input_size, embed_size))
            idx = 0

            for word in news:
                try:
                    vec = w2v.wv[word]
                    arr[idx] = vec
                    idx += 1
                    del vec
                    if idx == input_size:
                        break
                except KeyError:
                    pass
            batch_x_contents.append(arr)


        batch_x_titles = np.array(batch_x_titles)
        batch_x_titles = torch.from_numpy(batch_x_titles)
        batch_x_titles = batch_x_titles.view(-1, 1, input_size2, embed_size)
        batch_x_titles = batch_x_titles.type(torch.FloatTensor)

        batch_x_contents = np.array(batch_x_contents)
        batch_x_contents = torch.from_numpy(batch_x_contents)
        batch_x_contents = batch_x_contents.view(-1, 1, input_size, embed_size)
        batch_x_contents = batch_x_contents.type(torch.FloatTensor)

        batch_y = torch.LongTensor(batch_y).view(-1, 1)
        batch_y_one = torch.FloatTensor(len(batch_y), 2)
        batch_y_one.zero_()
        batch_y_one.scatter_(1, batch_y, 1)
        batch_y_one = batch_y_one.type(torch.LongTensor)

        batch_indi = torch.FloatTensor(batch_indi).view(-1, seq_len, 19)

        index += batch_size
        left = length - index
        i += 1

        yield i, batch_x_titles, batch_x_contents, batch_indi, batch_y_one, left, max_batch


def early_stopping_and_save_model(sess, saver, input_vali_loss, early_stopping_val_loss_list):

    if len(early_stopping_val_loss_list) != early_stopping_patience:
        early_stopping_val_loss_list = [99.99 for _ in range(early_stopping_patience)]

    early_stopping_val_loss_list.append(input_vali_loss)
    if input_vali_loss < min(early_stopping_val_loss_list[:-1]):
        saver.save(sess, "/home/ir1067/Final_Result/model/all_combined_{}_{}.ckpt".format(learning_rate, company))
        early_stopping_val_loss_list.pop(0)

        return True, early_stopping_val_loss_list

    elif early_stopping_val_loss_list.pop(0) < min(early_stopping_val_loss_list):
        return False, early_stopping_val_loss_list

    else:
        return True, early_stopping_val_loss_list

"""
TimeEffect(dic=out_['train'], w2v=w2v, batch_size=256,
                       labeling=1, embed_size=embed_size, input_size=input_size)

test_gen = TimeEffect(dic=data, w2v=w2v, batch_size=batch_size,
                           labeling=label, embed_size=embed_size, input_size=input_size)
"""

if __name__ == "__main__":
    model = Word2Vec.load('/home/ir1067/FOR_CONTENT/news_Contents_data_token/draft_mo.model')
    # model = Word2Vec.load('/home/ir1067/FOR_CONTENT/news_Contents_data_token/draft_mo22.model')

    w2v_size = 300
    min_word_num = 15
    max_word_num = 150  # 200 words
    lstm_cell_size = 50
    seq_len = 7
    input_dim = 19

    conv_l_1_node = 128
    # filter_size = [1, 2, 3, 4, 5]
    filter_size = [2]
    fc_l_1_node = 256
    fc_l_2_node = 4096
    fc_l_3_node = 256
    cg_reduction_size = 32
    spatial_gate_size = 2
    n_class = 2
    dropout_rate = 0.15
#    input_size = 150

    epoch_num = 200
    early_stopping_patience = 20
    batch_size = 4096
    learning_rates = [0.003, 0.005, 0.001, 0.002, 0.007, 0.01, 0.0005]
    # learning_rates = [0.005]
    learning_rate = 0.01
# =============================================================================

###############################################################################
#     create model
###############################################################################


df = pd.DataFrame()
for learning_rate in learning_rates:
    for num in range(len(list(data_new_vali2['train'].keys()))):
        tf.reset_default_graph()

        x_data = tf.placeholder(tf.float32, [None, max_word_num, w2v_size], name='x_data')
        x_data2 = tf.placeholder(tf.float32, [None, min_word_num, w2v_size], name='x_data2')
        x_indi = tf.placeholder(tf.float32, [None, seq_len, input_dim], name='x_indi')
        re_x_data = tf.reshape(x_data, [-1, max_word_num, w2v_size, 1], name='re_x_data')
        y_data = tf.placeholder(tf.float32, [None, 2], name='y_data')
        initializer = tf.contrib.layers.xavier_initializer()

        lstm = tf.nn.rnn_cell.LSTMCell(lstm_cell_size, activation=tf.nn.tanh, initializer=initializer)
        lstm2 = tf.nn.rnn_cell.LSTMCell(lstm_cell_size, activation=tf.nn.tanh, initializer=initializer)
        lstm_out, last_hidden = tf.nn.bidirectional_dynamic_rnn(lstm, lstm2, x_indi, dtype=tf.float32)
        lstm_out = tf.concat(axis=2, values=lstm_out)[:, -1, :]

        # LSTM - Self with title
        word_lstm = tf.nn.rnn_cell.LSTMCell(lstm_cell_size, initializer=initializer, name='3')
        word_lstm2 = tf.nn.rnn_cell.LSTMCell(lstm_cell_size, initializer=initializer, name='4')
        word_lstm_out, word_last_hidden = tf.nn.bidirectional_dynamic_rnn(word_lstm, word_lstm2, x_data2, dtype=tf.float32)
        word_lstm_out = tf.concat(axis=2, values=word_lstm_out)  # (batch size, 100, 1)
        final_state = word_lstm_out[:, -1, :][..., tf.newaxis]

        att_weight = tf.nn.softmax(tf.matmul(word_lstm_out, final_state))
        weighted_out = tf.tile(att_weight, tf.Variable([1, 1, lstm_cell_size * 2], dtype=tf.int32)) * word_lstm_out
        reshaped_word_vec = tf.reshape(weighted_out, (-1, lstm_cell_size * 2 * min_word_num))

        mod = sys.modules[__name__]
        for i in filter_size:
            setattr(mod, 'conv_w_{}'.format(i),
                    tf.Variable(tf.random_normal([i, w2v_size, 1, conv_l_1_node], stddev=0.01),
                                name='conv_w_{}'.format(i)))
            setattr(mod, 'conv_b_{}'.format(i), tf.Variable(tf.zeros([conv_l_1_node]), name='conv_b_{}'.format(i)))
            setattr(mod, 'conv_u_{}'.format(i),
                    tf.nn.conv2d(re_x_data, getattr(mod, 'conv_w_{}'.format(i)), strides=[1, 1, 1, 1], padding='VALID',
                                 name='conv_u_{}'.format(i)))
            setattr(mod, 'conv_z_{}'.format(i),
                    tf.nn.relu(tf.nn.bias_add(getattr(mod, 'conv_u_{}'.format(i)), getattr(mod, 'conv_b_{}'.format(i))),
                               name='conv_z_{}'.format(i)))
            # setattr(mod, 'conv_p_{}'.format(i), tf.nn.max_pool(getattr(mod, 'conv_z_{}'.format(i)), ksize=[1, max_word_num-i+1, 1, 1], strides=[1, max_word_num-i+1, 1, 1], padding='VALID', name='conv_p_{}'.format(i)))

        conv_out = getattr(mod, 'conv_z_{}'.format(filter_size[0]))

        cg_maxpool = tf.nn.max_pool(conv_out, ksize=[1, max_word_num - filter_size[0] + 1, 1, 1],
                                    strides=[1, max_word_num - filter_size[0] + 1, 1, 1], padding='VALID',
                                    name='cg_maxpool')
        cg_avgpool = tf.nn.avg_pool(conv_out, ksize=[1, max_word_num - filter_size[0] + 1, 1, 1],
                                    strides=[1, max_word_num - filter_size[0] + 1, 1, 1], padding='VALID',
                                    name='cg_avgpool')
        cg_maxpool = tf.reshape(cg_maxpool, [-1, conv_l_1_node])
        cg_avgpool = tf.reshape(cg_avgpool, [-1, conv_l_1_node])

        # channel gate reduction
        cg_reduce_w = tf.Variable(tf.random_normal([conv_l_1_node, cg_reduction_size], stddev=0.01), name='cg_reduce_w')
        cg_reduce_b = tf.Variable(tf.zeros([cg_reduction_size]), name='cg_reduce_b')
        cg_reduce_maxout = tf.nn.relu(
            tf.nn.bias_add(tf.matmul(cg_maxpool, cg_reduce_w, name='cg_reduce_maxmatmul'), cg_reduce_b),
            name='cg_reduce_maxout')
        cg_reduce_avgout = tf.nn.relu(
            tf.nn.bias_add(tf.matmul(cg_avgpool, cg_reduce_w, name='cg_reduce_avgmatmul'), cg_reduce_b),
            name='cg_reduce_avgout')

        # channel gate expand
        cg_expand_w = tf.Variable(tf.random_normal([cg_reduction_size, conv_l_1_node], stddev=0.01), name='cg_expand_w')
        cg_expand_b = tf.Variable(tf.zeros([conv_l_1_node]), name='cg_expand_b')
        cg_expand_maxout = tf.nn.bias_add(tf.matmul(cg_reduce_maxout, cg_expand_w, name='cg_expand_maxmatmul'),
                                          cg_expand_b)
        cg_expand_avgout = tf.nn.bias_add(tf.matmul(cg_reduce_avgout, cg_expand_w, name='cg_expand_avgmatmul'),
                                          cg_expand_b)

        # channel gate sum
        cg_att_raw = tf.nn.sigmoid(cg_expand_maxout + cg_expand_avgout, name='cg_att_sigmoid')
        cg_tile_tensor = tf.Variable([1, max_word_num - filter_size[0] + 1, 1, 1], dtype=tf.int32)
        cg_att_expand = tf.expand_dims(tf.expand_dims(cg_att_raw, 1), 1)
        cg_att_tile = tf.tile(cg_att_expand, cg_tile_tensor)

        # channel gate output
        conv_cg_out = conv_out * cg_att_tile

        # spatial gate
        sg_maxpool = tf.reduce_max(conv_cg_out, axis=3, keepdims=True, name='sg_maxpool')
        # sg_maxpool = tf.nn.max_pool(conv_cg_out, ksize=[1, 1, 1, conv_l_1_node], strides=[1, 1, 1, conv_l_1_node], padding='VALID', name='sg_maxpool')
        sg_avgpool = tf.reduce_mean(conv_cg_out, axis=3, keepdims=True, name='sg_avgpool')
        # sg_avgpool = tf.nn.avg_pool(conv_cg_out, ksize=[1, 1, 1, conv_l_1_node], strides=[1, 1, 1, conv_l_1_node], padding='VALID', name='sg_avgpool')
        sg_concat = tf.concat([sg_maxpool, sg_avgpool], axis=3, name='sg_concat')
        sg_conv_w = tf.Variable(tf.random_normal([spatial_gate_size, 1, 2, 1], stddev=0.01), name='sg_conv_w')
        sg_conv_b = tf.Variable(tf.zeros([1]), name='sg_conv_b')
        sg_conv_out = tf.nn.conv2d(sg_concat, sg_conv_w, strides=[1, 1, 1, 1], padding='SAME', name='sg_conv_out')
        sg_att_raw = tf.sigmoid(tf.nn.bias_add(sg_conv_out, sg_conv_b))
        sg_tile_tensor = tf.Variable([1, 1, 1, conv_l_1_node], dtype=tf.int32)
        sg_att_expand = tf.tile(sg_att_raw, sg_tile_tensor, name='sg_att_tile')
        conv_sg_out = conv_cg_out * sg_att_expand

        # conv_output = tf.concat([getattr(mod, 'conv_p_{}'.format(i)) for i in filter_size], axis=3, name='conv_output')
        re_conv_output = tf.reshape(conv_sg_out, [-1, (max_word_num - filter_size[0] + 1) * conv_l_1_node],
                                    name='re_conv_output')

        out_vec = tf.concat((reshaped_word_vec, lstm_out), axis=1, name='concat_conv_lstm')
        out_vec = tf.concat((re_conv_output, out_vec), axis=1, name='concat_conv_lstm2')

        # fc_w_1 = tf.Variable(tf.random_normal([conv_l_1_node*len(filter_size), fc_l_1_node], stddev=0.01), name='fc_w_1')
        # fc_b_1 = tf.Variable(tf.zeros([fc_l_1_node]), name='fc_b_1')
        # fc_bn_1 = tf.layers.batch_normalization(tf.matmul(re_conv_output, fc_w_1)+fc_b_1, name='fc_bn_1')
        # fc_z_1 = tf.nn.relu(fc_bn_1, name='fc_z_1')
        # fc_d_1 = tf.nn.dropout(fc_z_1, dropout_rate, name='fc_d_1')
        #
        # fc_w_2 = tf.Variable(tf.random_normal([fc_l_1_node, n_class], stddev=0.01), name='fc_w_2')
        # fc_u_2 = tf.matmul(fc_d_1, fc_w_2, name='fc_u_2')

        # fc_w_2 = tf.Variable(tf.random_normal([re_conv_output.shape[1].value, fc_l_2_node], stddev=0.01), name='fc_w_2')
        # fc_w_2 = tf.Variable(tf.random_normal([out_vec.shape[1].value, fc_l_2_node], stddev=0.01), name='fc_w_2')
        # fc_w_2 = tf.get_variable('fc_w_2', [out_vec.shape[1].value, fc_l_2_node], initializer=initializer)
        # fc_b_2 = tf.Variable(tf.zeros([fc_l_2_node]), name='fc_b_2')
        # # fc_bn_2 = tf.layers.batch_normalization(tf.matmul(re_conv_output, fc_w_2) + fc_b_2, name='fc_bn_2')
        # fc_bn_2 = tf.layers.batch_normalization(tf.matmul(out_vec, fc_w_2) + fc_b_2, name='fc_bn_2')
        # fc_z_2 = tf.nn.tanh(fc_bn_2, name='fc_z_2')
        # fc_d_2 = tf.nn.dropout(fc_z_2, dropout_rate, name='fc_d_2')

        fc_w_3 = tf.get_variable('concat_conv_lstm2', [out_vec.shape[1].value, n_class], initializer=initializer)
        fc_u_3 = tf.matmul(out_vec, fc_w_3, name='fc_u_3')

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc_u_3, labels=y_data), name='loss')
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train = optimizer.minimize(loss)

        pred_y = tf.nn.softmax(fc_u_3, name='pred_y')
        pred = tf.equal(tf.argmax(pred_y, 1), tf.argmax(y_data, 1), name='pred')
        acc = tf.reduce_mean(tf.cast(pred, tf.float32), name='acc')

        ###############################################################################
        #     model train
        ###############################################################################

        # i, train_x, y, left, tot = next(train_gen)

        gpu_options = tf.GPUOptions(visible_device_list="0")
        company_list = list(data_new_vali2['train'].keys())[num:num + 1]
        company = company_list[0]

        # batch_index_list = list(range(0, train_x.shape[0], batch_size))
        train_loss_list, vali_loss_list = [], []
        train_acc_list, vali_acc_list = [], []

        # start_time = time.time()
        # print('\n%s\n%s - training....'%('-'*100, save_model_name))
        # for op in [x_data, y_data, loss, acc, pred_y]:
        #     tf.add_to_collection(tf_model_important_var_name, op)
        saver = tf.train.Saver()
        early_stopping_val_loss_list = []
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())

            early_stop = 0

            for epoch in range(epoch_num):
                train_gen = TimeEffect(dic=data_new_vali2['train'], w2v=model, batch_size=batch_size, embed_size=300,
                                       input_size=max_word_num, input_size2=min_word_num, num=num)
                test_gen = TimeEffect(dic=data_new_vali2['vali'], w2v=model, batch_size=batch_size, embed_size=300,
                                      input_size=max_word_num, input_size2=min_word_num, num=num)

                total_loss, total_acc, vali_acc, vali_loss = 0, 0, 0, 0
                processing_bar_var = [0, 0]

                for i, batch_x_title, batch_x_content, batch_indi, batch_y, left, tot in train_gen:
                    if not i % 150:
                        print('{} / {}'.format(i, tot))
                    batch_x_title = np.reshape(batch_x_title.numpy(), (-1, min_word_num, 300))
                    batch_x_content = np.reshape(batch_x_content.numpy(), (-1, max_word_num, 300))
                    batch_y = batch_y.numpy()

                    sess.run(train, feed_dict={x_data2: batch_x_title, x_data: batch_x_content, x_indi: batch_indi, y_data: batch_y})
                    # sess.run(train, feed_dict={x_data: batch_x, y_data: batch_y})
                    loss_val, acc_val = sess.run([loss, acc],
                                                 feed_dict={x_data2: batch_x_title, x_data: batch_x_content, x_indi: batch_indi, y_data: batch_y})
                    # loss_val, acc_val = sess.run([loss, acc],
                    #                              feed_dict={x_data: batch_x, y_data: batch_y})

                    total_loss += loss_val
                    total_acc += acc_val

                train_loss_list.append(total_loss / tot)
                train_acc_list.append(total_acc / tot)

                print('\n#%4d/%d' % (epoch + 1, epoch_num), end='  |  ')
                print('Avg_loss={:.4f} / Avg_acc={:.4f}'.format(total_loss / tot, total_acc / tot), end='  |  \n')

                for i, vali_x_title, vali_x_content, vali_indi, vali_y, left, tot in test_gen:
                    # for i, vali_x, vali_y, left, tot in test_gen:
                    vali_x_title = np.reshape(vali_x_title.numpy(), (-1, min_word_num, 300))
                    vali_x_content = np.reshape(vali_x_content.numpy(), (-1, max_word_num, 300))
                    vali_y = vali_y.numpy()

                    tmp_vali_loss, tmp_vali_acc, vali_pred_y = sess.run([loss, acc, pred_y],
                                                                        feed_dict={x_data2: vali_x_title, x_data: vali_x_content, x_indi: vali_indi,
                                                                                   y_data: vali_y})
                    # feed_dict = {x_data: vali_x,
                    #              y_data: vali_y})

                    print('{} / {}\t{} / {}'.format(i, tot, np.argmax(vali_pred_y, axis=1).sum(), len(vali_pred_y)))

                    vali_loss += tmp_vali_loss
                    vali_acc += tmp_vali_acc

                vali_loss_list.append(vali_loss / tot)
                vali_acc_list.append(vali_acc / tot)

                # tmp_running_time = time.time() - start_time
                # print('\n#%4d/%d' % (epoch + 1, epoch_num), end='  |  ')
                # print('Avg_loss={:.4f} / Avg_acc={:.4f}'.format(total_loss/tot, total_acc/tot), end='  |  ')
                print('\t')
                print('epoch {}: vali_loss={:.4f} / vali_acc={:.4f}'.format(epoch + 1, vali_loss / tot, vali_acc / tot),
                      end='  |  \n\n')

                bool_continue, early_stopping_val_loss_list = early_stopping_and_save_model(sess, saver,
                                                                                            vali_loss_list[-1],
                                                                                            early_stopping_val_loss_list)
                if not bool_continue:
                    print('{0}\nstop epoch : {1}\n{0}'.format('-' * 100, epoch - early_stopping_patience + 1))
                    break

            print('-' * 100)

            tmp = pd.DataFrame([company, train_loss_list, train_acc_list, vali_loss_list, vali_acc_list,
                                np.argmax(vali_pred_y, axis=1).sum(), len(vali_pred_y), max(vali_acc_list)])
            df = pd.concat([df, tmp])

            # saver = tf.train.Saver()
            # save_file = saver.save(sess, "/home/ir1067/Result/model/text_w_indi_LSTM_{}_{}.ckpt".format(learning_rate, company))
        df.to_csv("/home/ir1067/Final_Result/all_combined_{}.csv".format(learning_rate))


###############################################################################
#     model test
###############################################################################
for learning_rate in learning_rates:
    ttest_loss = []
    ttest_acc = []
    company_list = list(data_new_vali2['train'].keys())
    for num, company in enumerate(list(data_new_vali2['train'].keys())):
        tf.reset_default_graph()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.import_meta_graph(
                "/home/ir1067/Final_Result/model/all_combined_{}_{}.ckpt".format(learning_rate, company) + '.meta')
            # saver.restore(sess, tf.train.latest_checkpoint('./'))
            saver.restore(sess, "/home/ir1067/Final_Result/model/all_combined_{}_{}.ckpt".format(learning_rate, company))

            test_gen = TimeEffect(dic=data_new_vali2['test'], w2v=model, batch_size=batch_size, embed_size=300,
                                  input_size=max_word_num, input_size2=min_word_num, num=num)

            x_data = tf.get_default_graph().get_tensor_by_name('re_x_data:0')
            x_data2 = tf.get_default_graph().get_tensor_by_name('x_data2:0')
            x_indi = tf.get_default_graph().get_tensor_by_name('x_indi:0')
            y_data = tf.get_default_graph().get_tensor_by_name('y_data:0')
            acc = tf.get_default_graph().get_tensor_by_name('acc:0')
            loss = tf.get_default_graph().get_tensor_by_name('loss:0')
            pred_y = tf.get_default_graph().get_tensor_by_name('pred_y:0')
            pred = np.array([])
            prob = np.array([])
            for i, test_x_title, test_x_content, test_indi, test_y, left, tot in test_gen:
                test_loss, test_acc, test_pred_y, test_true_y = sess.run([loss, acc, pred_y, y_data],
                                                                         feed_dict={
                                                                             x_data2: test_x_title.reshape(-1, min_word_num, 300),
                                                                             x_data: test_x_content.reshape(-1, max_word_num, 300, 1),
                                                                             x_indi: test_indi,
                                                                             y_data: test_y})

                y_prob = np.max(test_pred_y, axis=1)
                y_true = np.argmax(test_true_y, axis=1)
                y_pred = np.argmax(test_pred_y, axis=1)
                len(test_true_y)
                # print(classification_report(y_true, y_pred, target_names=['Positive', 'Negative']))
                print(pd.crosstab(pd.Series(y_true), pd.Series(y_pred), rownames=['True'], colnames=['Predicted'],
                                  margins=True))
                print('\ntest_loss = {:.4f} / test_acc = {:.4f}\n{:s}'.format(test_loss, test_acc, '=' * 100))
                pred = np.append(pred, y_pred)
                prob = np.append(prob, y_prob)
            final_y = pd.DataFrame({'y_pred' : pred, 'y_prob' : prob})
            ttest_loss.append(test_loss)
            ttest_acc.append(test_acc)
            final_y.to_csv('/home/ir1067/Final_Result/pred_y/pred_y_all_combined_{}_{}.csv'.format(company, learning_rate))
    test_result = pd.DataFrame({'company' : company_list, 'test_loss' : ttest_loss, 'test_acc' : ttest_acc})
    test_result.to_csv('/home/ir1067/Final_Result/test_result/all_combined_{}_{}.csv'.format(company, learning_rate))


for learning_rate in learning_rates:

    accuracy_0 = []
    accuracy_1 = []
    include_accuracy = []
    softmax_accuracy = []
    length = []
    ratio_0 = []
    ratio_1 = []
    ratio_inc = []
    ratio_sof = []
    company_list = list(data_new_vali2['train'].keys())
    for company in list(data_new_vali2['train'].keys()):
        final_y = pd.read_csv('/home/ir1067/Final_Result/pred_y/pred_y_all_combined_{}_{}.csv'.format(company, learning_rate), engine='python',
                              index_col=0)
        # final_y.columns = ['pred']
        final_x = pd.read_excel('/home/ir1067/For_Train_All_7/{}_train.xlsx'.format(company))
        final_x.index = final_x['dates']
        final_x = final_x['2018-01-01':'2018-12-31']
        final_x = final_x.reset_index(drop=True)
        final_data = pd.concat([final_x, final_y], axis=1)

        final_data = final_data[['dates', 'label', 'y_pred', 'y_prob']]

        dates = final_data['dates'].unique()
        final_data.index = final_data['dates']

        acc_0 = 0
        acc_1 = 0
        softmax_acc = 0
        equal = 0
        include_acc = 0
        for first, second in zip(dates[:-1], dates[1:]):
            tmp_data = final_data[first:first]
            tmp = 0
            for i in range(len(tmp_data)):
                if tmp_data.iloc[i]['label'] == tmp_data.iloc[i]['y_pred']:
                    tmp += 1

            if tmp == (len(tmp_data) / 2):
                include_acc += 1
                if tmp_data[tmp_data['y_pred'] == 1]['y_prob'].sum() > tmp_data[tmp_data['y_pred']==0]['y_prob'].sum():
                    if tmp_data.iloc[0]['label'] == 1:
                        softmax_acc += 1
                else:
                    if tmp_data.iloc[0]['label'] == 0:
                        softmax_acc += 1

                if tmp_data.iloc[0]['label'] == 1:
                    acc_1 += 1
                if tmp_data.iloc[0]['label'] == 0:
                    acc_0 += 1

            if tmp > (len(tmp_data) / 2):
                include_acc += 1
                softmax_acc += 1
                acc_0 += 1
                acc_1 += 1

        print(company, include_acc, '/', len(dates), round(include_acc / len(dates) * 100, 2), '\n',\
              company, softmax_acc, '/', len(dates), round(softmax_acc / len(dates) * 100, 2), '\n',\
              company, acc_0, '/', len(dates), round(acc_0 / len(dates) * 100, 2), '\n',\
              company, acc_1, '/', len(dates), round(acc_1 / len(dates) * 100, 2))
        # print(company, final_acc-equal,'/', len(dates)-equal, round((final_acc-equal)/(len(dates)-equal) *100, 2))
        accuracy_0.append(acc_0)
        accuracy_1.append(acc_1)
        include_accuracy.append(include_acc)
        softmax_accuracy.append(softmax_acc)
        length.append(len(dates))
        ratio_0.append(round(acc_0 / len(dates) * 100, 2))
        ratio_1.append(round(acc_1 / len(dates) * 100, 2))
        ratio_inc.append(round(include_acc / len(dates) * 100, 2))
        ratio_sof.append(round(softmax_acc / len(dates) * 100, 2))
    zippedList = list(zip(company_list, accuracy_0, length, ratio_0, accuracy_1, length, ratio_1, include_accuracy, length, ratio_inc, softmax_accuracy, length, ratio_sof))
    result = pd.DataFrame(zippedList, columns=['company', 'accuracy_0', 'length', 'ratio', 'accuracy_1', 'length', 'ratio', 'accuracy_inc', 'length', 'ratio', 'accuracy_sof', 'length', 'ratio'])
    result.to_csv('/home/ir1067/Final_Result/result_all_combined_{}.csv'.format(learning_rate))

#
# ###############################################################################
# #     predict label
# ###############################################################################
#     os.chdir(model_test_data_file_path)
#     try:
#         with open(model_test_data_no_label_file_name[1], 'rb') as f:
#             model_test_x, _ = pickle.load(f)
#     except:
#         model_test_x, _ = load_data_fn(model_test_data_no_label_file_name[0], w2v_model, w2v_size, max_word_num, model_test_data_no_label_file_name[1])
#     print('model_test_shape : {}'.format(model_test_x.shape))
#
#     tf.reset_default_graph()
#     os.chdir(save_model_path)
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#
#         saver = tf.train.import_meta_graph(tf_model_path+'.meta')
#         saver.restore(sess, tf_model_path)
#         x_data, y_data, loss, acc, pred_y = tf.get_collection(tf_model_important_var_name)
#
#         model_y_pred = np.argmax(sess.run(pred_y, feed_dict={x_data: model_test_x}), axis=1)
#
#     os.chdir(pred_label_file_path)
#     pd.DataFrame({'pred_label':model_y_pred}).to_csv(pred_label_file_name, index=False)
#
# ###############################################################################
# #     predict acc
# ###############################################################################
#     os.chdir(model_test_data_file_path)
#     model_y_true = pd.read_csv(model_test_data_true_file_name, encoding='cp949')['label'].values
#
#     for per_name in ['ywy', '김동우', '김준호', '민은주', '정상형', '이지현']:
#         if per_name == 'ywy':
#             os.chdir(pred_label_file_path)
#         else:
#             os.chdir(hw_ai_file_path)
#
#         tmp_pred_label_file_name = pred_label_file_name.replace('ywy', per_name)
#         print('%s\n%s\n'%('#'*100, tmp_pred_label_file_name))
#
#         try:
#             model_y_pred = pd.read_csv(tmp_pred_label_file_name, engine='python')['pred_label'].values
#
#             print(classification_report(model_y_true, model_y_pred, target_names=['Positive', 'Negative']))
#             print(pd.crosstab(pd.Series(model_y_true), pd.Series(model_y_pred), rownames=['True'], colnames=['Predicted'], margins=True))
#
#             model_test_acc = np.sum(np.equal(model_y_true, model_y_pred)) / np.shape(model_y_true)[0]
#             print('\nmodel_test_acc = {:.4f}'.format(model_test_acc))
#         except FileNotFoundError as e:
#             print(e)
#         except ValueError as e:
#             print(e)
#     print('#'*100)
#