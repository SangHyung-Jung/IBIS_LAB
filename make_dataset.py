import os
import numpy as np
import pandas as pd
from util.preprocess import NewsProcessor
from gensim.models import Word2Vec
import datetime
from util.indicator import add_tech_stats


# make technical indicator
def mk_TA(load_path="/home/ir1067/price_data/", save_path="/home/ir1067/price_w_indicator/"):
    for company in os.listdir(load_path):
        if company == "유한양행.xlsx":
            pass
        else:
            continue
        print(company + 'start')
        add_tech_stats(company[:-5], save_path)


# build word2vec model
def save_token_and_make_w2v(news_path='/home/ir1067/news_Contents_33', save_path='./news_w2v.model', token_save_path="/home/ir1067/title_data_token/"):
    text = []

    for company in sorted(os.listdir(news_path))[6:7]:
        print('{} start'.format(company[:-17]))
        data = pd.read_excel(news_path + '/' + company)
        processor.clean_text(data)
        data['tokenized'] = processor.tokenizing(data=data, tag=tag)
        processor.drop_empty(data)

        data.to_excel(os.path.join(token_save_path, '{}_token.xlsx'.format(company[:-17])))

        for i in range(len(data)):
            text.append(data['tokenized'][i])
        print('{} complete'.format(company))

    word2vec_model = Word2Vec(text, size=300, window=3, min_count=1, workers=1, sg=1)
    word2vec_model.save(save_path)
    print('W2V save.\ndone.')


# save label & indicator
def save_label_and_TA(indicator_path='/home/ir1067/indicator/', label_path='/home/ir1067/label_data/'):

    for company in processor.company_list:
        try:
            data = processor.get_csv(company)
            label_data = processor.labeling(data, processor.kospi, days=[1, 5, 20, 30, 60])
            label_data = label_data[label_data.columns[-5:]]
            ta_data = label_data[label_data.columns[:-5]]
            label_data.to_csv(os.path.join(label_path, '{}_label.csv'.format(company)))
            ta_data.to_csv(os.path.join(indicator_path, '{}_indicator.csv'.format(company)))
        except ValueError as e:
            print(company)
            print(e)

    print("Label & indicator save done")


# Binding tokenized text with label and indicator
def make_dataset(token_path='/home/ir1067/title_data_token/',
                label_path='/home/ir1067/label_data/',
                indicator_path='/home/ir1067/indicator/',
                save_path='/home/ir1067/token_indicator_label/'):

    dates = processor.kospi.index.strftime("%Y-%m-%d")

    for company in processor.company_list:
        print(company, "start!")
        data = pd.read_excel(os.path.join(token_path, '{}_token.xlsx'.format(company), index_col=0))
        try:
            data.drop(index=datetime.datetime(2015, 1, 1), inplace=True)
        except KeyError:
            pass

        print("length: " + str(len(data)))
        label = pd.read_csv(os.path.join(label_path, '{}_label.csv'.format(company), index_col=0))
        label.index = pd.to_datetime(label.index)
        l = []
        indicator_out = []
        indicator_list = ['rsi_14', 'cci', 'stoch_slowk', 'stoch_slowd', 'willr', 'momentum', 'roc',
                        'Sto_rsi_fastk', 'Sto_rsi_fastd', 'adosc']
        indicator = pd.read_csv(os.path.join(indicator_path, '{}_indicator.csv'.format(company), index_col=0))[indicator_list]
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
        output.to_excel(os.path.join(save_path, '{}.xlsx'.format(company)))
        print('{} done'.format(company))


if __name__ == '__main__':
    processor = NewsProcessor()
    tag = ['NNG', 'NNP', 'NNB', 'NP', 'NP', 'NR',
        'VV', 'VA', 'MM', 'MAG', 'MAJ', 'SL', 'SH']
    mk_TA()
    save_token_and_make_w2v()
    save_label_and_TA()
    make_dataset()
