###模型的输入参数的格式为[batch_size, max_seq_len],
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from transformers import BertModel


class Bert_GCN_Att(nn.Module):
    def __init__(self, rela,gcn_drop,class_drop, hidden_dim=768 ,bias=True):
        super(Bert_GCN_Att, self).__init__()
        self.tagset_size = len(rela)
        # # self.hidden = self.init_hidden()
        # self.lstm = nn.LSTM(bidirectional=True, num_layers=2, input_size=768+200, hidden_size=(hidden_dim+200) // 2,
        #                     batch_first=True)
        # self.dropout = nn.Dropout(0.5)

        self.hidden_dim = hidden_dim
        self.sent_len = 50
        self.batch_size = 16
        self.pos_emd=nn.Embedding(15,200,padding_idx=0)

        self.bert = BertModel.from_pretrained('F:\项目\模型\chinese_wobert_plus_L-12_H-768_A-12')

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.dense = nn.Linear(self.hidden_dim*3, self.tagset_size)
        # self.loss = nn.CrossEntropyLoss(ignore_index=0)  # 忽略pad的损失
        self.gcn_drop1 = nn.Dropout(gcn_drop)
        # self.gcn_drop2 = nn.Dropout(0.5)
        self.class_dropout = nn.Dropout(class_drop)

        # gcn
        # self.weight1 = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
        # self.weight2= nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
        self.gcn_linear1 = nn.Linear(hidden_dim+200, hidden_dim)
        # self.gcn_linear2 = nn.Linear(hidden_dim, hidden_dim)
        # self.gcn_linear3 = nn.Linear(hidden_dim+200, hidden_dim)
        # if bias:
        #     self.bias1 = nn.Parameter(torch.FloatTensor(hidden_dim))
        #     self.bias2 = nn.Parameter(torch.FloatTensor(hidden_dim))
        # else:
        #     self.register_parameter('bias', None)

        self.att_linear = nn.Linear(self.sent_len, self.sent_len)
        self.entity_linear = nn.Linear(self.sent_len, self.sent_len)
        # self.att_drop = nn.Dropout(0.5)
        # self.entity_drop = nn.Dropout(0.3)
        # self.reset_parameters()


    def reset_parameters(self):
        init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        if self.bias1 is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight1)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias1, -bound, bound)
        init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        if self.bias2 is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight2)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias2, -bound, bound)

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _bert_enc(self, x):
        """
        x: [batchsize, sent_len]
        enc: [batch_size, sent_len, 768]
        """
        with torch.no_grad():
            encoded_layer = self.bert(x)
            enc = encoded_layer[0]
        return enc

    def _get_lstm_features(self, embeds):
        """sentence is the ids"""
        # self.hidden = self.init_hidden()
        # embeds = self._bert_enc(sentence)  # enc: [batch_size, sent_len, 768]
        # 过lstm
        enc, _ = self.lstm(embeds)
        # dropout
        lstm_feats = self.dropout(enc)
        # lstm_feats = self.fc(enc)
        return lstm_feats  # [batch_size, sent_len, hidden_dim]

    def gcn(self, sentence, adj,linear,drop):
        Ax = adj.float().bmm(sentence)
        AxW = linear(Ax)
        denom = torch.sum(adj, dim=2, keepdim=True)  # 无需+1，adj设置对角为1
        AxW = AxW/denom + linear(sentence) # self loop类似于残差链接，防止过拟合
        gAxW = torch.tanh(AxW)

        gcn_output = drop(gAxW)

        return gcn_output

    def gcn_new(self, text, adj):
        hidden = torch.matmul(text.float(), self.weight1.float())
        denom = torch.sum(adj, dim=2, keepdim=True)
        output1 = torch.matmul(adj.float(), hidden) / denom
        output1 = output1 + self.bias1
        output1=self.gcn_drop(torch.tanh(output1))

        hidden = torch.matmul(output1.float(), self.weight2.float())
        denom = torch.sum(adj, dim=2, keepdim=True)
        output2 = torch.matmul(adj.float(), hidden) / denom
        output2 = output2 + self.bias2
        output2 = self.gcn_drop2(torch.tanh(output2))
        return output2

    def attention(self, H):
        batch_size, max_len, feat_dim = H.shape

        M = H.transpose(1, 2)  # [batch_size, hidden_dim,sent_len]
        score = torch.bmm(H, M)  # [batch_size,sent_len,sent_len]
        # score = torch.bmm(score, att_weight)  # [batch_size,sent_len,sent_len
        # score=att_weight(score)
        result = F.softmax(score, dim=2)
        return result

    def pos_pool(self, H, pos):#提取指定行的向量

        pos = pos.eq(0).unsqueeze(2)

        h_pos = H.masked_fill(pos, -100000)  # 将非指定位置填充为极小数

        h_pos = torch.max(h_pos, 1)[0]  # 在第一维最大池化，维度变为BxS，取出实体行对应值

        return h_pos


    def loss(self,entity_att,word_pos,rela_pos,out,y):
        pos_loss = torch.mean(1 - torch.cosine_similarity(entity_att * word_pos, rela_pos))
        class_loss_fuc = nn.CrossEntropyLoss()
        class_loss = class_loss_fuc(out, y)  # 已经求过平均了
        return class_loss+pos_loss


    def forward(self, sentence, adj, entity_pos1, entity_pos2,valid_pos,pos):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        word_embed = self._bert_enc(sentence)# [batch_size, sent_len, hidden_dim]

        # lstm_feats = self._get_lstm_features(word_embed)  # [batch_size, sent_len, hidden_dim]
        # lstm_att = self.attention(lstm_feats)  # [batch_size, sent_len, sent_len]

        # gcn
        pos_emb = self.pos_emd(pos)
        embeds = torch.cat([word_embed, pos_emb], -1)
        # lstm_feats = self._get_lstm_features(embeds)  # [batch_size, sent_len, hidden_dim]


        gcn_feats = self.gcn(embeds, adj,self.gcn_linear1,self.gcn_drop1)
        # embeds=torch.cat([gcn_feats,pos_emb],-1)
        # gcn_feats = self.gcn(gcn_feats, adj,self.gcn_linear2,self.gcn_drop2)
        # gcn_feats=torch.cat([gcn_feats,pos_emb],-1)

        gcn_att = self.attention(gcn_feats)  # [batch_size, sent_len, sent_len]

        # att = lstm_att+gcn_att #B*S*S
        att = self.att_linear(gcn_att)
        # H=torch.bmm(lstm_att,lstm_feats)+torch.bmm(gcn_att,gcn_feats)
        H = torch.bmm(gcn_att, gcn_feats)
        # H=gcn_feats

        entity_att = self.pos_pool(att, entity_pos1)+self.pos_pool(att, entity_pos2)
        entity_att = self.entity_linear(entity_att)

        rela_att=entity_att*valid_pos
        rela_pos = (rela_att == rela_att.max(dim=1, keepdim=True)[0]).to(dtype=torch.int32)

        r_h=self.pos_pool(H, rela_pos)
        e1_h = self.pos_pool(H, entity_pos1)
        e2_h = self.pos_pool(H, entity_pos2)
        out=torch.cat([r_h,e1_h,e2_h],dim=1)
        out = self.dense(self.class_dropout(out))
        # out = self.dense(out)

        return entity_att,out
        # return score, tag_seq
