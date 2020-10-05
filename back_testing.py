import os
import pandas as pd
import numpy as np
import pickle
import sklearn.metrics as metrics

company = ['CJ대한통운',
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

def backtesting(path, data_path, result_path, save_path, company=company):
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

    # calculating precision recall
    precision = []
    recall = []
    f1_score = []
    accuracy = []
    company_list = company.copy()
    ff = ['CNN_T', 'LSTM_T', '7_w_softmax_LSTM_LSTM', '7_w_softmax_TITLE', 'CNN_T_CNN_C', 'LSTM_T_LSTM_C', '+CNN_T_LSTM_C', 'all_combined']
    for file_name in ff:
        precision = []
        recall = []
        f1_score = []
        accuracy = []
        for company in company_list:
            final_y = pd.read_csv(os.path.join(result_path, file_name+'_{}_{}.csv'.format(company, '0.005')), engine='python',
                                    index_col=0)

            final_x = pd.read_excel(os.path.join(data_path, '{}_train.xlsx'.format(company)))
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
            for first, _ in zip(dates[:-1], dates[1:]):
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
                final_pred_ys.append(final_pred_y)

            final = pd.DataFrame(list(zip(final_labels, final_pred_ys)), columns = ['label', 'y_pred'])
            acc = metrics.accuracy_score(final.label, final.y_pred)
            pre = metrics.precision_score(final.label, final.y_pred)
            re = metrics.recall_score(final.label, final.y_pred)
            f1 = metrics.f1_score(final.label, final.y_pred)

            accuracy.append(acc)
            precision.append(pre)
            recall.append(re)
            f1_score.append(f1)

        finals = pd.DataFrame(list(zip(company_list, accuracy, precision, recall, f1_score)), columns=['company', 'accuracy', 'precision', 'recall', 'f1_score'])
        # save backtesting result
        finals.to_csv(os.path.join(save_path, '{}_{}.csv'.format(file_name, '0.005')), encoding= 'EUC-KR')
