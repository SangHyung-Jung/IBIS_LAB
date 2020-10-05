from gensim.models.word2vec import Word2Vec
import torch.nn as nn
import torch
from util.data import load_data_dict, data_cleaning, dataloader
from util.model import MainModel


def train():
    # load w2v model
    w2v = Word2Vec.load(w2v_path)

    # process data
    data = load_data_dict([('train', '2015-01-01', '2017-06-30'),
                            ('vali', '2017-07-01', '2018-06-30'),
                            ('test', '2018-07-01', '2018-12-31')])

    data = data_cleaning(data,
                        rm_word=['코스피', '공시', '코스닥', '종목', '주요', 'KOSPI', 'Kospi', '헤드라인', '헤드 라인'],
                        rm_count=50,
                        contain_name=True)

    loader = dataloader(data, w2v, batch_size, w2v_size, contents_size, title_size)

    model = MainModel(title_size, lstm_cell_size, conv_channel, w2v_size, att_code_size, indicator_len, contents_size, fc_hiddens)
    model.cuda()

    criterion = nn.BCELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(1, epoch_num + 1):
        for i, batch_x_titles, batch_x_contents, batch_indi, batch_y_one, left, max_batch in loader:
            optimizer.zero_grad()
            out = model(batch_x_titles, batch_x_contents, batch_indi)
            loss = criterion(out, batch_y_one)
            loss.backward()
            optimizer.step()
    

if __name__ == "__main__":
    w2v_path = '/home/ir1067/FOR_CONTENT/news_Contents_data_token/draft_mo.model'

    w2v_size = 300
    title_size = 15
    contents_size = 150  # 200 words
    lstm_cell_size = 50
    indicator_len = 19
    conv_channel = 128
    fc_hiddens = [512, 256]
    att_code_size = 32

    epoch_num = 200
    batch_size = 4096
    learning_rate = 0.001

    train()
