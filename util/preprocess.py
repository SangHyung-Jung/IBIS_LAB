import os
import pandas as pd
import numpy as np
import re
from khaiii import KhaiiiApi


class NewsProcessor:
    def __init__(self):
        # .xlsx file = news text data
        self.xlsx_list = [name for name in os.listdir('/home/ir1067/FOR_TITLE/Title_33_2014_all') if ('.xlsx' in name) & ('#' not in name)]
        self.xlsx_list.sort()

        # company names
        self.company_list = set([file[:-5] for file in self.xlsx_list])

        # .csv file = financial data [open, close, price]
        self.csv_list = [name for name in os.listdir('/home/ir1067/price_w_indicator') if ('.csv' in name) & ('#' not in name)]
        self.csv_list.sort()

        # kospi market data [open, close]
        self.kospi = pd.read_csv("/home/ir1067/data/kospi.csv").set_index('date')
        self.kospi.index = pd.to_datetime(self.kospi.index)
        self.kospi = self.kospi[['open', 'close']]
        self.kospi['open'] = [re.sub(',', '', text) for text in self.kospi['open']]
        self.kospi['close'] = [re.sub(',', '', text) for text in self.kospi['close']]
        self.kospi.open = self.kospi.open.astype(float)
        self.kospi.close = self.kospi.open.astype(float)
        # Khaiii API
        self.khaiii = KhaiiiApi()

        # print info
        print("Data file infomation")
        print("- News data (xlsx):\t{}".format(len(self.xlsx_list)))
        print("- Price data (csv):\t{}".format(len(self.csv_list)))
        print("- Company count:\t{}".format(len(self.company_list)))

    def get_xlsx(self, company_name):
        output = pd.DataFrame()

        # xlsx files containing company name
        data_list = [filename for filename in self.xlsx_list if company_name in filename]

        for filename in data_list:
            news = pd.read_excel(filename, index_col=0)
            output = pd.concat([output, news])
        output.reset_index(inplace=True, drop=True)

        print("Data NaN info")
        for col in output.columns:
            output[col] = [text if text != "" else np.nan for text in output[col]]
        print(output.isna().sum())

        output['date'] = pd.to_datetime(output['date'], format="%Y.%m.%d")
        output.set_index('date', drop=True, inplace=True)

        return output

    def get_csv(self, company_name):
        output = pd.DataFrame()

        # csv files containing company name
        data_list = [filename for filename in self.csv_list if company_name in filename]

        for filename in data_list:
            price = pd.read_csv('/home/ir1067/price_w_indicator/' + filename, index_col=0)
            output = pd.concat([output, price])

        output.index = pd.to_datetime(output.index, format="%Y.%m.%d")

        return output


    def clean_text(self, data):
        # ⓒ~ , 저작권자~, 기자~ 삭제 고려해보기

        data['title'] = [re.sub('\[.+?\]', '', text, 0, re.I | re.S).strip() \
                                for text in data['title']]
        data['title'] = [text if text != '' else np.nan for text in data['title']]

        data['contents'] = [text.replace("// flash 오류를 우회하기 위한 함수 추가\nfunction _flash_removeCallback() {}", "") \
                            for text in data['contents']]
        data['contents'] = [re.sub('\(.+?\)', '', text, 0, re.I | re.S).strip() \
                            for text in data['contents']]
        data['contents'] = [re.sub('{.+?}', '', text, 0, re.I | re.S).strip() \
                            for text in data['contents']]
        data['contents'] = [re.sub('\[.+?\]', '', text, 0, re.I | re.S).strip() \
                            for text in data['contents']]
        data['contents'] = [re.sub('<.+?>', '', text, 0, re.I | re.S).strip() \
                            for text in data['contents']]
        data['contents'] = [re.sub('＜.+?＞', '', text, 0, re.I | re.S).strip() \
                            for text in data['contents']]
        # ▶ 이걸로 시작하는 기사가 하나 있음
        #data['contents'] = [re.sub('▶.*', '', text, 0, re.I | re.S).strip().replace(",", "") \
        #                     for text in data['contents']]

        print("Check NaN")
        print(data.isna().sum())

    def drop_empty(self, data):
        print("Data length before drop: ", len(data))
        data.dropna(inplace=True, how='any')
        data.drop(data[data['contents'] == ''].index, inplace=True)
        data = data.reset_index(drop=True)
        #index = [news.index for news in data['contents'] if news == '']
        #for i in index:
        #   data.drop([data.index[i]], inplace= True)
        print("Data length after drop:  ", len(data))


    def tokenizing(self, data, tag):
        if type(tag) == list:
            try:
                print("Start Khaiii analyze")
                after_analyze = [self.khaiii.analyze(news) for news in data['contents'] if news != '']
                print("Done")

                tokenized = [[morph.lex for chunk in news for morph in chunk.morphs \
                       if morph.tag in tag] for news in after_analyze]
                tokenized = [text if text != [] else np.nan for text in tokenized]

                is_empty = [1 if text == np.nan else 0 for text in tokenized]
                print("Empty list after tokenizing: {}".format(sum(is_empty)))

                return tokenized

            except KeyError as e:
                print(e, "DataFrame does not have 'contents' column")
        else:
            print("Error: parameter 'tag' must be list")


    def tokenizing_title(self, data, tag):
        if type(tag) == list:
            try:
                print("Start Khaiii analyze")
                after_analyze = [self.khaiii.analyze(news) for news in data['title'] if news != '']
                print("Done")

                tokenized_title = [[morph.lex for chunk in news for morph in chunk.morphs \
                       if morph.tag in tag] for news in after_analyze]
                tokenized_title = [text if text != [] else np.nan for text in tokenized_title]

                is_empty = [1 if text == np.nan else 0 for text in tokenized_title]
                print("Empty list after tokenizing: {}".format(sum(is_empty)))

                return tokenized_title

            except KeyError as e:
                print(e, "DataFrame does not have 'contents' column")
        else:
            print("Error: parameter 'tag' must be list")


    def labeling(self, data, kospi, days):
        #data = pd.read_csv('/home/ir1067/price_w_indicator/유한양행.csv', index_col=0)
        #kospi = pd.read_csv('/home/ir1067/data/kospi.csv', index_col=0)
        data.index = pd.to_datetime(data.index)
        #data = data.loc[self.kospi.index]

        #data = data.loc[kospi.index]
        kospi.columns = ['k_open', 'k_close']
        data = pd.merge(data, kospi, how='right', left_index=True, right_index=True)
        data = data.dropna(how='any')

        for day in days:
            if day == 1:
                open = data['open']
                close = data['close']
                rtn = close / open - 1
                mkt_rtn = data['k_close'] / data['k_open'] - 1
                data['label'] = (rtn > mkt_rtn).astype(int).shift(-1)
            else:
                price = data['adj_close']
                rtn = price.pct_change(day).shift(-day - 1)
                mkt_rtn = data['k_close'].pct_change(day).shift(-day - 1)
                data['label%d' % day] = (rtn > mkt_rtn).astype(int)

        data.drop(['k_open', 'k_close'], inplace=True, axis=1)
        indicators = data[data.columns[:]]

        return indicators

    def to_datetime(self, unified_file):
        for i in range(len(unified_file.date)):
            if i % 100 == 0:
                print('processing', i)
            if len(unified_file.date[i]) == 19:
                unified_file.date[i] = unified_file.date[i][:15] + '0' + unified_file.date[i][15:]

            if bool(re.search('오후', unified_file.date[i])) & bool(unified_file.date[i][15:17] != '12') == True:
                unified_file.date[i] = unified_file.date[i][:15] + '{}'.format(int(unified_file.date[i][15:17]) + 12) + \
                                        unified_file.date[i][17:]

            unified_file.date[i] = unified_file.date[i][:12] + unified_file.date[i][15:]

        unified_file.date = pd.to_datetime(unified_file.date)
        unified_file.set_index(['date'], inplace=True)
