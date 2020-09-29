## TO교정 ##
'''
이 py에서는
저기 preprocess py 사용해서
온라인 뉴스 데이터 전처리 하는 식으로 하는 것들 모아놓음
뭐가 너무 많은데 필요없는거는 깔끔하게 지우고 해도 될듯

그리고 이 py의 결과물로 내가 보내준 zip 파일의 train data들이 나오는게 좋을꺼 같아용
'''


import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.preprocess import NewsProcessor
from gensim.models import Word2Vec
import pickle
import gzip
import datetime
import random
from collections import Counter
import copy
from datetime import timedelta
from tqdm import tqdm

# set data path
# data_path = "/home/ir1067/data/"
# os.chdir(data_path)

processor = NewsProcessor()
# build word2vec model ##########################
text = []
tag = ['NNG', 'NNP', 'NNB', 'NP', 'NP', 'NR',
       'VV', 'VA', 'MM', 'MAG', 'MAJ', 'SL', 'SH']
data_path = '/home/ir1067/news_Contents_33'
for company in sorted(os.listdir(data_path))[6:7]:
    print('{} start'.format(company[:-17]))
    data = pd.read_excel(data_path + '/' + company)
    processor.clean_text(data)
    data['tokenized'] = processor.tokenizing(data=data, tag=tag)
    processor.drop_empty(data)

    data.to_excel('/home/ir1067/news_Contents_data_token/{}_token.xlsx'.format(company[:-17]))
    print('{} done'.format(company))

    for i in range(len(data)):
        text.append(data['tokenized'][i])
    print('{} complete'.format(company))

word2vec_model = Word2Vec(text, size=300, window=3, min_count=1, workers=1, sg=1)
word2vec_model.save('draft_mo.model')

model = Word2Vec.load('draft_mo.model')

################################################

# word2vec pretrained
# model = Word2Vec.load("word2vec_model.bin")

################################################

# Target data label 따로 저장 (label_data 폴더)
for company in processor.company_list:
    try:
        data = processor.get_csv(company)
        label_data = processor.labeling(data, processor.kospi, days=[1, 5, 20, 30, 60])
        label_data = label_data[label_data.columns[-5:]]
        label_data.to_csv('/home/ir1067/label_data/{}_label.csv'.format(company))
    except ValueError as e:
        print(company)
        print(e)

####################################################################

# indicator만 따로 저장 (indicator 폴더)
for company in processor.company_list:
    try:
        data = processor.get_csv(company)
        label_data = processor.labeling(data, processor.kospi, days=[1, 5, 20, 30, 60])
        label_data = label_data[label_data.columns[:-5]]
        label_data.to_csv('/home/ir1067/indicator/{}_indicator.csv'.format(company))
    except ValueError:
        print(company)

#####################################################################
# Khaii로 토큰화 진행해서 저장  ## 이거를 W2V모델 생성할때 저장해놨어야 되는데 안했다 개빡친
# ['tokenized'] column 새로 생성해서 title_data_token 폴더에 저장// csv로 하고 싶었는데 cp949도 이상한 단어 많아서 그런지 에러나서 그냥 엑셀로 함
# 참고로 지금 토큰파일 저장할 때 Excel 한 파일에 들어갈 수있는 url이 65350개라서 삼성전자처럼 큰것들은 UserWarning 뜸
for company in processor.company_list:
    print('{} start'.format(company))
    data = processor.get_xlsx(company)
    processor.clean_text(data)
    processor.drop_empty(data)
    data['tokenized'] = processor.tokenizing(data=data, tag=tag)
    processor.drop_empty(data)


####################################################################

####################################################################
# Title Combine to one List

X = []

for company in processor.company_list:
    print('{} start'.format(company))
    data = pd.read_excel('/home/ir1067/title_data_token/{}_token.xlsx'.format(company))
    X += list(data['tokenized'])

X_to_list = [each.replace("[", "").replace("]", "").replace("'", "").split(", ") for each in X]
X_to_text = [''.join(text) for text in X_to_list]
tmp = [text for text in X_to_text if
       not ('코스피' in text) | ('공시' in text) | ('코스닥' in text) | ('종목' in text) | ('이 시각' in text) \
           | ('주요' in text) | ('KOSPI' in text)]

tokenize_cnt = Counter(tmp)
tokenize_cnt.most_common(10000)[-100:]
tokenize_ = [text for text, cnt in tokenize_cnt.items() if cnt < 11]
Counter(tokenize_)
tokens = [each.replace("[", "").replace("]", "").replace("'", "").split(",") for each in d['tokenized']]

X_rm_kospi = [text for text in X if
              not ('코스피' in text) | ('공시' in text) | ('코스닥' in text) | ('종목' in text) | ('이 시각' in text) \
                  | ('주요' in text) | ('KOSPI' in text)]
[text for text in X if type(text) == int]
set(X).__len__()

sentence_count = Counter(X)
cnt = Counter(X_rm_kospi)
sentence_count.most_common(100)

tmp = [text for text, cnt in cnt.items() if cnt < 11]
len(tmp)

##################################################################

# Binding tokenized text with label

##################################################################
dates = processor.kospi.index.strftime("%Y-%m-%d")
for company in processor.company_list:
    print(company, "start!")
    data = pd.read_excel('/home/ir1067/title_data_token/{}_token.xlsx'.format(company), index_col=0)
    try:
        data.drop(index=datetime.datetime(2015, 1, 1), inplace=True)
    except KeyError:
        pass

    print("length: " + str(len(data)))
    label = pd.read_csv('/home/ir1067/label_data/{}_label.csv'.format(company), index_col=0)
    label.index = pd.to_datetime(label.index)
    l = []

    for first, second in zip(dates[:-1], dates[1:]):
        cropped = data[first: second]['tokenized']
        if len(cropped):
            try:
                cropped = cropped.drop(index=datetime.datetime(int(second[:4]), int(second[5:7]), int(second[8:10])))
            except KeyError as e:
                print(e)
            try:
                tmp = list(label.loc[first, :])
                l += [tmp for _ in range(len(cropped))]
            except KeyError:
                l += [[np.nan, np.nan, np.nan, np.nan, np.nan] for _ in range(len(cropped))]

    y = pd.DataFrame(l, columns=['label1', 'label5', 'label20', 'label30', 'label60'])
    y.index = data.index
    output = pd.concat([data, y], axis=1)
    output.dropna(inplace=True, how='any')

    print("I/O...")
    output.to_excel('/home/ir1067/token_label/{}.xlsx'.format(company))
    print('{} done'.format(company))

##################################################################

# Binding tokenized text with label and indicator

##################################################################
dates = processor.kospi.index.strftime("%Y-%m-%d")
for company in processor.company_list:
    print(company, "start!")
    data = pd.read_excel('/home/ir1067/title_data_token/{}_token.xlsx'.format(company), index_col=0)
    try:
        data.drop(index=datetime.datetime(2015, 1, 1), inplace=True)
    except KeyError:
        pass

    print("length: " + str(len(data)))
    label = pd.read_csv('/home/ir1067/label_data/{}_label.csv'.format(company), index_col=0)
    label.index = pd.to_datetime(label.index)
    l = []
    indicator_out = []
    indicator_list = ['rsi_14', 'cci', 'stoch_slowk', 'stoch_slowd', 'willr', 'momentum', 'roc',
                      'Sto_rsi_fastk', 'Sto_rsi_fastd', 'adosc']
    indicator = pd.read_csv('/home/ir1067/indicator/{}_indicator.csv'.format(company), index_col=0)[indicator_list]
    indicator.index = pd.to_datetime(indicator.index)

    for first, second in zip(dates[:-1], dates[1:]):
        cropped = data[first: second]['tokenized']
        if len(cropped):
            try:
                cropped = cropped.drop(index=datetime.datetime(int(second[:4]), int(second[5:7]), int(second[8:10])))
            except KeyError as e:
                print(e)
            try:
                tmp_label = list(label.loc[first, :])
                tmp_indicator = list(indicator.loc[first, :])
                l += [tmp_label for _ in range(len(cropped))]
                indicator_out += [tmp_indicator for _ in range(len(cropped))]
            except KeyError:
                l += [[np.nan, np.nan, np.nan, np.nan, np.nan] for _ in range(len(cropped))]
                indicator_out += [[np.nan] * 10 for _ in range(len(cropped))]

    y = pd.DataFrame(l, columns=['label1', 'label5', 'label20', 'label30', 'label60'])
    indicator_df = pd.DataFrame(indicator_out, columns=indicator_list)
    y.index = data.index
    indicator_df.index = data.index
    output = pd.concat([data, indicator_df, y], axis=1)
    output.dropna(inplace=True, how='any')

    print("I/O...")
    output.to_excel('/home/ir1067/token_indicator_label/{}.xlsx'.format(company))
    print('{} done'.format(company))


def get_xy(date_list):
    output = dict()
    os.chdir("/home/ir1067/token_indicator_label/")
    files = os.listdir()
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


data_new_vali = get_xy([('train', '2015-01-01', '2017-06-30'),
                        ('vali', '2017-07-01', '2018-06-30'),
                        ('test', '2018-07-01', '2018-12-31'))

output['train']['현대차']['indicator'][0]
"""
            output[tvt][company] = dict()
            output[tvt][company]['text'] = tokens

            for label in ['label1', 'label5', 'label20', 'label30', 'label60']:
                try:
                    output[tvt][company][label] += list(d[label])
                except KeyError:
                    output[tvt][company][label] = []
                    output[tvt][company][label] += list(d[label])

    return output
"""

output = get_xy()

# Save data
# with gzip.open('/home/ir1067/Data_rm_over50_DropDupl_containName.pickle', 'wb') as f:
#    pickle.dump(data_dict, f)

all_text = []
for tvt in ['train', 'test', 'vali']:
    for
com in data_dict[tvt].keys():
all_text += data_dict[tvt][com]['text']

data_dict['train']['현대중공업']['text'].__len__()
data_dict['train']['현대중공업']['label1'].__len__()

print(len(all_text))

all_label = []
for tvt in ['test']:
    for
com in data_dict[tvt].keys():
for label in ['label1']:
    all_label += data_dict[tvt][com][label]

sum(all_label) / len(all_label)

# check empty text
all_text = []
for tvt in ['train', 'test', 'vali']:
    for
com in data_dict[tvt].keys():
if data_dict[tvt][com]['text'] == []:
    print(tvt, com)

all_label.__len__()

print(len(all_label['label5']))


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
    for company in dictionary['train'].keys():
        out['train'][company]['text'] = [sentence for sentence in out_['train'][company]['text'] if
                                         not word in sentence]

    for company in dictionary['vali'].keys():
        out['vali'][company]['text'] = [sentence for sentence in out_['vali'][company]['text'] if not word in sentence]

    for company in dictionary['test'].keys():
        out['test'][company]['text'] = [sentence for sentence in out_['test'][company]['text'] if not word in sentence]

    return out


df = pd.DataFrame(out_['train']['LG화학'])

df['merge_text'] = [''.join(text) for text in df['text']]
df['merge_text']

all_text = []
for tvt in ['train', 'vali', 'test']:
    for company in out_[tvt].keys():
        all_text += out_[tvt][company]['text']
all_text[0]

t = df.drop(df[df['merge_text'] == '본준새고민LG전자TV사업부진'].index)
t.head()
df.index[df['merge_text'].str.contains('코스피', regex=False).tolist()]
df['merge_text'].index
t = df.drop_duplicates('merge_text')
df.to_dict('list')['label1']


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
    # print("text_over_count: %d" % len(text_over_count))
    # text_under_count = {text:(False if cnt == 1 else True) for text, cnt in text_cnt.items()}

    print("for loop")
    # let's remove
    for tvt in ['train', 'vali', 'test']:
        for company in outdict[tvt].keys():
            tmp = pd.DataFrame(outdict[tvt][company])
            b_len = len(tmp)

            tmp['merge_text'] = [''.join(text) for text in tmp['text']]

            # get only cnt 1 sentence:
            rm_sen_b = len(tmp)
            tmp.drop([idx for idx, text in zip(tmp['merge_text'].index, tmp['merge_text']) if text in text_over_count],
                     inplace=True)
            # tmp.drop([idx for idx, text in zip(tmp['merge_text'].index, tmp['merge_text']) if text_under_count[text]],
            #          inplace=True)
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

            # if company != "SK지주":
            #    tmp['text'] = [[word for word in text if word != company] for text in tmp['text']]
            # else:
            #    tmp['text'] = [[word for word in text if word != 'SK'] for text in tmp['text']]

            tmp4 = len(tmp)
            tmp.drop([i for i, text in tmp['text'].iteritems() if len(text) == 1], inplace=True)
            only_name = tmp4 - len(tmp)

            leng = len(tmp)
            # tmp.drop([idx for idx, l in zip(tmp['text'].index, tmp['text']) if l == []], inplace=True)

            outdict[tvt][company] = tmp.drop('merge_text', axis=1).to_dict('list')

            print(tvt, company,
                  "\trm sentence: {}, rm word: {}, rm nest: {}, rm only name: {}, length: {}".format(rm_sen,
                                                                                                     rm_word_len,
                                                                                                     rm_nest,
                                                                                                     only_name,
                                                                                                     leng))
            # print(tvt, company, "\trm empty list: {}".format(leng - len(tmp)))

    return outdict


# out_contain_name = copy.deepcopy(out_clean)

data_dict = data_cleaning(data_dict,
                          rm_word=['코스피', '공시', '코스닥', '종목', '주요', 'KOSPI', 'Kospi', '헤드라인', '헤드 라인'],
                          rm_count=50,
                          contain_name=True)

a = pd.DataFrame(data_dict['train']['현대차'])

sum(out_clean['train']['현대차']['label5']) / len(out_clean['train']['현대차']['label5'])

all_text = []
label1 = []
label5 = []
all_text = []
for tvt in ['train', 'vali', 'test']:
    for company in out_contain_name[tvt].keys():
        # all_text += out_contain_name[tvt][company]['text']
        # label5 += out_contain_name[tvt][company]['label5']
        # label1 += out_contain_name[tvt][company]['label1']

        all_text += [word for text in out_contain_name[tvt][company]['text'] for word in text]
a = Counter(all_text)
a.most_common(40)

for i, (text, l1, l5) in enumerate(zip(all_text, label1, label5)):
    print(text, l1, l5)
    if i == 20:
        break

t = [''.join(text) for text in all_text]
t[-20:]

out_rm_kospi = remove_word(out_, '코스피')
out_rm_kospi_kosdaq = remove_word(out_rm_kospi, '코스닥')

text_train = []
text_vali = []
text_test = []

for company, text_dic in out_rm_kospi_kosdaq['train'].items():
    text_train += text_dic['text']
for company, text_dic in out_rm_kospi_kosdaq['vali'].items():
    text_vali += text_dic['text']
for company, text_dic in out_rm_kospi_kosdaq['test'].items():
    text_test += text_dic['text']

text_all = text_train + text_vali + text_test
text_all.__len__()
words = [word for sentence in text_all for word in sentence]

words.__len__()

# word_cnt = Counter(words)
word_rm_kospi = Counter(words)
word_rm_kospi_kosdaq = Counter(words)

word_cnt.most_common(40)
word_rm_kospi.most_common(40)
word_rm_kospi_kosdaq.most_common(40)

word_rm_kospi['코스닥']
word_rm_kospi_kosdaq['코스닥']

word_rm_kospi.__len__()
word_cnt.__len__()

text_train.__len__()
text_vali.__len__()
text_test.__len__()

# ==============================================================================
# get indicator mean, std
# ==============================================================================

label_list = os.listdir('/home/ir1067/label_data')
company_list = [company[:-10] for company in label_list]

all_data = merge_csv(company_list)
all_data.index = pd.to_datetime(all_data.index)
indi_mean = all_data['2015':'2017-06'][X_train_rm.columns].mean()
indi_std = all_data['2015':'2017-06'][X_train_rm.columns].std()

m = indi_mean.values
s = indi_std.values
m = m.reshape((1, -1))

mm = np.array([m, m])
mm + m
