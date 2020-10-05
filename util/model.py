import torch
import torch.nn as nn
import torch.nn.functional as F

class MainModel(nn.Module):
    def __init__(self, lstm_input, lstm_hidden, conv_out, w2v_size, att_code, indicator_len, content_len, fc_hidden):
        self.lstm_hidden = lstm_hidden

        self.lstm = nn.LSTM(input_size=lstm_input, hidden_size=lstm_hidden, num_layer=1, bidirectional=True)
        self.conv = nn.Conv2d(in_channels=1, out_channels=conv_out, kernel_size=(3, w2v_size), padding=1, stride=1)
        self.spatial_att = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(3, 1), padding=1, stride=1)
        self.avg_channel_att_e = nn.Linear(in_features=conv_out, out_features=att_code)
        self.avg_channel_att_d = nn.Linear(in_features=att_code, out_features=att_code)
        self.max_channel_att_e = nn.Linear(in_features=conv_out, out_features=att_code)
        self.max_channel_att_d = nn.Linear(in_features=att_code, out_features=att_code)
        self.fc1 = nn.Linear(in_features=lstm_hidden+indicator_len+conv_out*content_len, out_features=fc_hidden[0])
        self.fc2 = nn.Linear(in_features=fc_hidden[0], out_features=fc_hidden[1])
        self.classifier = nn.Linear(fc_hidden[1], 1)

    def forward(self, title, content, indicator):
        # title
        lstm_out, hidden = self.lstm(title)
        attention_value = torch.sigmoid(torch.matmul(lstm_out, hidden.permute(1, 2, 0)))
        weighted_lstm_out = attention_value.repeat((1, 1, self.lstm_hidden)) * lstm_out
        title_out = weighted_lstm_out.mean(dim=1) # size -> lstm_hidden

        # contents
        # channel attention
        conv_out = self.conv(content).squeeze(3)  # B, C, H
        _, C, H = conv_out.size()
        channel_max = conv_out.max(dim=2)
        channel_avg = conv_out.mean(dim=2)
        channel_max_att = F.sigmoid(self.max_channel_att_d(F.sigmoid(self.max_channel_att_e(channel_max)))).unsqeeze(2)
        channel_avg_att = F.sigmoid(self.avg_channel_att_d(F.sigmoid(self.avg_channel_att_e(channel_avg)))).unsqeeze(2)
        conv_out = channel_max_att.repeat((1,1,H)) * conv_out
        conv_out = channel_avg_att.repeat((1,1,H)) * conv_out
        # spatial attention
        spatial_max = conv_out.max(dim=1, keepdim=True)
        spatial_avg = conv_out.mean(dim=1, keepdim=True)
        spatial = torch.cat((spatial_max, spatial_avg), dim=1)
        spatial_att = F.sigmoid(self.spatial_att(spatial))
        conv_out = spatial_att.repeat((1, C, 1)) * conv_out

        # concat all
        out = torch.cat((title_out, conv_out.view(-1, C*H), indicator))

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))

        return F.sigmoid(self.classifier(out))
