'''
아마 내가 막판에 손 좀 봤을 텐데
main이랑 합쳐서 해도 될듯하고
아님 Class로 만들어서 main에서 같이 실행되게 해도 될꺼같고 그럼
'''

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from tqdm import tqdm
companys = ['CJ대한통운',
'GS건설',
'KAKAO',
'KB금융',
'KT&G',
'LG생활건강',
'LG전자',
'LG화학',
'NAVER',
'POSCO',
'S-Oil',
'SKT',
'SK이노베이션',
'SK하이닉스',
'고려아연',
'기아차',
'대림산업',
'대우조선해양',
'삼성생명',
'삼성전자',
'삼성중공업',
'셀트리온',
'신한지주',
'에스원',
'한국전력',
'한미약품',
'현대건설',
'현대글로비스',
'현대모비스',
'현대제철',
'현대중공업',
'현대차']

# data path
path = r"G:\내 드라이브\ibislab\dd\\"

# date data load
with open(path+"dates.pickle", 'rb') as f:
    dates = pickle.load(f)

# kospi data load
kospi = pd.read_csv(path+"kospi.csv", index_col=0)
kospi.index = pd.to_datetime(kospi.index)  # datetime index
kospi = kospi['2018']  # test period
kopen = kospi['open'][0]  # kopen -> first open price
accrtn = (kospi[['close']] - kopen) / kopen  # get accumulate return of kospi
accrtn.columns = ['kospi_acc']

# load prices & add daily return columns on DF
for com in company.copy():
    kia = pd.read_csv(path+com+".csv", engine='python', index_col=0)
    kia.index = pd.to_datetime(kia.index)
    kia = kia['2018']
    rtn = (kia['close'] - kia['open']) / kia['open']
    if (np.inf == rtn).sum() > 0:
        print(com)
        company.remove(com)
    else:
        accrtn[com+'_rtn'] = (kia['close'] - kia['open']) / kia['open']

# all combined pred merge
for com in company:
    ac_kia = pd.read_csv(path+"ac_"+com+'_'+'0.003'+".csv", engine='python', index_col=0)
    ac_kia['date'] = dates[com]
    ac_kia_g = ac_kia.groupby("date")['y_pred'].mean()
    ac_kia_g.index = pd.to_datetime(ac_kia_g.index)
    ac_kia_g.name = com+'_ac_pred'
    accrtn = pd.merge(accrtn, pd.DataFrame(ac_kia_g), left_index=True, right_index=True, how='left')

# back-testing
ac_acc = pd.DataFrame()
for com in company:
    tmp = []
    for rtn, pred in accrtn[[com+'_rtn', com+'_ac_pred']].values:
        if pred > 0.5:
            tmp.append(rtn + 1 - 0.003)
        else:
            tmp.append(1)
    ac_acc[com+'_acc'] = tmp
ac_acc.index = accrtn.index

# count wrong prediction
(ac_acc < 1).sum()

ac_acc = ac_acc.cumprod()
ac_acc -= 1
#ac_acc.to_csv(path+"test.csv")

accrtn['all_combined'] = ac_acc.mean(axis=1)


# LSTM T pred merge
for com in company:
    ac_kia = pd.read_csv(path+"lt_"+com+'_'+'0.003'+".csv", engine='python', index_col=0)
    ac_kia['date'] = dates[com]
    ac_kia_g = ac_kia.groupby("date")['y_pred'].mean()
    ac_kia_g.index = pd.to_datetime(ac_kia_g.index)
    ac_kia_g.name = com+'_lt_pred'
    accrtn = pd.merge(accrtn, pd.DataFrame(ac_kia_g), left_index=True, right_index=True, how='left')

for com in company:
    print(com, (accrtn[com+'_lt_pred'] > 0.5).sum())
for com in company:
    print(com, (accrtn[com+'_ac_pred'] > 0.5).sum())

# back-testing
lt_acc = pd.DataFrame()
for com in company:
    tmp = []
    for rtn, pred in accrtn[[com+'_rtn', com+'_lt_pred']].values:
        if pred > 0.5:
            tmp.append(rtn + 1 - 0.003)
        else:
            tmp.append(1)
    lt_acc[com+'_acc'] = tmp
lt_acc.index = accrtn.index
# count wrong prediction
(lt_acc < 0).sum()

lt_acc = lt_acc.cumprod()
lt_acc -= 1
accrtn['LSTM_T'] = lt_acc.mean(axis=1)

lt_acc.tail(10)
accrtn['한미약품_lt_pred'].tail(30)
accrtn['한미약품_rtn'].tail(30)
lt_acc.plot()
accrtn[['kospi_acc', 'all_combined', 'LSTM_T']].plot()
lt_acc.isna().sum()

## calculating precision recall

softmax_accuracy = []
length = []
ratio_0 = []
precision = []
recall = []
f1_score = []
accuracy = []
company_list = companys
ff = ['CNN_T', 'LSTM_T', '7_w_softmax_LSTM_LSTM', '7_w_softmax_TITLE', 'CNN_T_CNN_C', 'LSTM_T_LSTM_C', '+CNN_T_LSTM_C', 'all_combined']
for file_name in ff:
    precision = []
    recall = []
    f1_score = []
    accuracy = []
    for company in company_list:
        final_y = pd.read_csv(r'G:\내 드라이브\ibislab\final_result/pred_y/pred_y_'+file_name+'_{}_{}.csv'.format(company, '0.005'), engine='python',
                                index_col=0)
        # final_y.columns = ['pred']
        final_x = pd.read_excel(r'G:\내 드라이브\ibislab\revision\For_Train_All_7/{}_train.xlsx'.format(company))
        final_x.index = final_x['dates']
        final_x = final_x['2018-01-01':'2018-12-31']
        final_x = final_x.reset_index(drop=True)
        final_data = pd.concat([final_x, final_y], axis=1)

        final_data = final_data[['dates', 'label', 'y_pred', 'y_prob']]

        dates = final_data['dates'].unique()
        final_data.index = final_data['dates']
        
        softmax_acc = 0
        final_labels = []
        final_pred_ys = []
        for first, second in zip(dates[:-1], dates[1:]):
            tmp_data = final_data[first:first]
            tmp = 0

            final_label = tmp_data.label.median()
            final_pred_y = tmp_data.y_pred.median()

            for i in range(len(tmp_data)):
                if tmp_data.iloc[i]['label'] == tmp_data.iloc[i]['y_pred']:
                    tmp += 1

            if tmp == (len(tmp_data) / 2):
                if tmp_data[tmp_data['y_pred'] == 1]['y_prob'].sum() > tmp_data[tmp_data['y_pred']==0]['y_prob'].sum():
                    final_pred_y = 1
                    if tmp_data.iloc[0]['label'] == 1:
                        softmax_acc += 1
                        
                else:
                    final_pred_y = 0
                    if tmp_data.iloc[0]['label'] == 0:
                        softmax_acc += 1

            elif tmp > (len(tmp_data) / 2):
                softmax_acc += 1

            else:
                pass
            final_labels.append(final_label)
        print(company, file_name, final_labels.__len__())
            final_pred_ys.append(final_pred_y)

        final = pd.DataFrame(list(zip(final_labels, final_pred_ys)), columns = ['label', 'y_pred'])
        acc = metrics.accuracy_score(final.label, final.y_pred)
        pre = metrics.precision_score(final.label, final.y_pred) # precision : 1이라고 예측한 값 중 진짜 1인 확률
        re = metrics.recall_score(final.label, final.y_pred) # recall : 전체 1 중에서 내가 1이라고 예측한 확률
        f1 = metrics.f1_score(final.label, final.y_pred) # 평균

        accuracy.append(acc)
        precision.append(pre)
        recall.append(re)
        f1_score.append(f1)
    break
    finals = pd.DataFrame(list(zip(company_list, accuracy, precision, recall, f1_score)), columns=['company', 'accuracy', 'precision', 'recall', 'f1_score'])
    finals.to_csv(r'G:\내 드라이브\ibislab\revision\final_result\{}_{}.csv'.format(file_name, '0.005'), encoding= 'EUC-KR')
        # print(company, softmax_acc, '/', len(dates), round(softmax_acc / len(dates) * 100, 2))
        # print(company, final_acc-equal,'/', len(dates)-equal, round((final_acc-equal)/(len(dates)-equal) *100, 2))
    #     softmax_accuracy.append(softmax_acc)
    #     length.append(len(dates))
    #     ratio_sof.append(round(softmax_acc / len(dates) * 100, 2))
    # zippedList = list(zip(company_list, softmax_accuracy, length, ratio_sof))
    # result = pd.DataFrame(zippedList, columns=['company', 'accuracy_sof', 'length', 'ratio'])
    # result.to_csv('/home/ir1067/Final_Result/result_'+file_name+'_{}.csv'.format('0.003'))
