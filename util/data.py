import os
import pandas as pd
import random
from collections import Counter
import torch
import numpy as np


# load data to dictionary form
def load_data_dict(date_list, data_path="/home/ir1067/token_indicator_label/"):
    output = dict()
    os.chdir()
    files = os.listdir(data_path)
    output['train'] = dict()
    output['vali'] = dict()
    output['test'] = dict()

    for file in files:
        company = file.split('.')[0]
        print(company, "start!")
        data = pd.read_excel(file, index_col=0)
        print("I/O done")
        for tvt, start, end in date_list:
            d = data[start: end]

            d['text'] = [each.replace("[", "").replace("]", "").replace("'", "").split(", ") for each in d['tokenized']]
            d['indicator'] = [list(row) for i, row in
                              d[['rsi_14', 'cci', 'stoch_slowk', 'stoch_slowd', 'willr', 'momentum', 'roc',
                                 'Sto_rsi_fastk', 'Sto_rsi_fastd', 'adosc']].iterrows()]

            output[tvt][company] = d[
                ['text', 'indicator', 'label1', 'label5', 'label20', 'label30', 'label60']].to_dict('list')

    return output


# get stats
def stats(dic):
    avg_news_cnt = 0
    avg_news_length = 0
    label1 = 0
    label5 = 0
    label20 = 0
    label30 = 0
    label60 = 0

    for key, value in dic.items():
        avg_news_cnt += len(value['text'])
        for each in value['text']:
            avg_news_length += len(each)
        try:
            label1 += sum(value['label1'])
            label5 += sum(value['label5'])
            label20 += sum(value['label20'])
            label30 += sum(value['label30'])
            label60 += sum(value['label60'])
        except TypeError:
            print(value['label1'].index('label1'))

    print("avg news count: ", str(avg_news_cnt / 55))
    print("avg news length: ", str(avg_news_length / avg_news_cnt))
    print("label1 ratio: ", str(label1 / avg_news_cnt))
    print("label5 ratio: ", str(label5 / avg_news_cnt))
    print("label20 ratio: ", str(label20 / avg_news_cnt))
    print("label30 ratio: ", str(label30 / avg_news_cnt))
    print("label60 ratio: ", str(label60 / avg_news_cnt))


def data_len(dic):
    avg_news_cnt = 0

    for key, value in dic.items():
        avg_news_cnt += len(value['text'])

    return avg_news_cnt


def remove_word(dictionary, word):
    out = dictionary.copy()
    out_ = dictionary.copy()
    for company in dictionary['train'].keys():
        out['train'][company]['text'] = [sentence for sentence in out_['train'][company]['text'] if
                                         not word in sentence]

    for company in dictionary['vali'].keys():
        out['vali'][company]['text'] = [sentence for sentence in out_['vali'][company]['text'] if not word in sentence]

    for company in dictionary['test'].keys():
        out['test'][company]['text'] = [sentence for sentence in out_['test'][company]['text'] if not word in sentence]

    return out


def data_cleaning(dictionary, rm_word, rm_count, contain_name=False):
    all_text = []
    outdict = dictionary.copy()

    print("binding all text")
    for tvt in ['train', 'vali', 'test']:
        for company in outdict[tvt].keys():
            all_text += outdict[tvt][company]['text']

    all_merge_text = [''.join(text) for text in all_text]
    text_cnt = Counter(all_merge_text)

    print("count")
    text_over_count = [text for text, cnt in text_cnt.items() if cnt >= rm_count]

    print("for loop")
    # start removing
    for tvt in ['train', 'vali', 'test']:
        for company in outdict[tvt].keys():
            tmp = pd.DataFrame(outdict[tvt][company])
            b_len = len(tmp)

            tmp['merge_text'] = [''.join(text) for text in tmp['text']]

            # get only cnt 1 sentence:
            rm_sen_b = len(tmp)
            tmp.drop([idx for idx, text in zip(tmp['merge_text'].index, tmp['merge_text']) if text in text_over_count],
                     inplace=True)
            rm_sen = rm_sen_b - len(tmp)

            # remove word
            tmp2 = len(tmp)
            try:
                for word in rm_word:
                    tmp.drop(tmp.index[tmp['merge_text'].str.contains(word, regex=False).tolist()],
                             inplace=True)
            except AttributeError as e:
                print(e)
                print(tmp['text'])
            rm_word_len = tmp2 - len(tmp)

            # remove nested sentence
            tmp3 = len(tmp)
            tmp.drop_duplicates('merge_text', inplace=True)
            rm_nest = tmp3 - len(tmp)

            # contain name
            if contain_name:
                try:
                    if company != 'SK지주':
                        tmp = tmp[tmp['merge_text'].str.contains(company, regex=False)]
                    else:
                        tmp = tmp[tmp['merge_text'].str.contains('SK', regex=False)]
                except AttributeError:
                    pass


            tmp4 = len(tmp)
            tmp.drop([i for i, text in tmp['text'].iteritems() if len(text) == 1], inplace=True)
            only_name = tmp4 - len(tmp)

            leng = len(tmp)

            outdict[tvt][company] = tmp.drop('merge_text', axis=1).to_dict('list')

            print(tvt, company,
                  "\trm sentence: {}, rm word: {}, rm nest: {}, rm only name: {}, length: {}".format(rm_sen,
                                                                                                     rm_word_len,
                                                                                                     rm_nest,
                                                                                                     only_name,
                                                                                                     leng))
            # print(tvt, company, "\trm empty list: {}".format(leng - len(tmp)))

    return outdict


def dataloader(dic, w2v, batch_size, embed_size, input_size, input_size2, seq_len=7, companies=None, num=0):
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
        raise ValueError("'Companies' must be None or list")

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