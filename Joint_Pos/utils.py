# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2019/11/07 22:11:33
@Author  :   Cao Shuai
@Version :   1.0
@Contact :   caoshuai@stu.scu.edu.cn
@License :   (C)Copyright 2018-2019, MILAB_SCU
@Desc    :   None
'''

import os

import numpy as np
import logging
import torch
from torch.utils.data import Dataset
import numpy as np
import networkx as nx
from typing import Tuple, List
# from pytorch_pretrained_bert import BertTokenizer
from transformers import BertTokenizer

logger = logging.getLogger(__name__)

# bert_model = '/root/workspace/qa_project/chinese_L-12_H-768_A-12'
# bert_model = 'bert-base-chinese'
bert_model = 'F:\项目\模型\chinese_wobert_plus_L-12_H-768_A-12'
tokenizer = BertTokenizer.from_pretrained(bert_model)
# VOCAB = ('<PAD>','[CLS]', '[SEP]', 'O', 'R')
VOCAB = ('<PAD>', '[CLS]', '[SEP]',
         'B-PER', 'I-PER',
         'B-PLA', 'I-PLA',
         'B-EQU', 'I-EQU',
         'B-TIM', 'I-TIM',
         'B-ORG', 'I-ORG',
         'B-LOC', 'I-LOC',
         'B-TASK', 'I-TASK',
         'B-Accident', 'I-Accident',
         'R-involved_person', 'R-happen_time',
         'R-involved_org', 'R-located_in',
         'R-work_at', 'R-happen_place',
         'R-workmate', 'R-execute',
         'R-use', 'R-put', 'R-hired_by',
         'R-belong_to', 'O',)
# VOCAB = ('<PAD>', '[CLS]', '[SEP]', 'O', 'B-INF', 'I-INF', 'B-PAT', 'I-PAT', 'B-OPS', 'I-OPS', 'B-DSE', 'I-DSE', 'B-DRG', 'I-DRG', 'B-LAB', 'I-LAB')
pos2idx={ '91': 14,'92':1, '86':2, '85':3, '94':4, '95':5, '84':6,
         '100':7, '93':8, '90':9, '89':10, '96':11, '97':12, '101':13}

rela = ['R-execute', "R-work_at", "R-workmate", "R-use", "R-put", "R-involved_person",
     'R-happen_place','R-involved_org','R-happen_time','R-located_in','R-hired_by','R-belong_to']
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}
MAX_LEN = 50-2


class NerDataset(Dataset):
    def __init__(self, f_path):
        with open(f_path, 'r', encoding='utf-8') as fr:
            entries = fr.read().strip().split('\n\n')
        sents, tags_li, heads ,poss= [], [], [] ,[] # list of lists
        for entry in entries:
            words = [line.split()[0] for line in entry.splitlines()]
            tags = [line.split()[1] for line in entry.splitlines()]
            head_id = [int(line.split()[2]) for line in entry.splitlines()]
            pos = [line.split()[-1] for line in entry.splitlines()]
            #head_id 从0开始算
            heads.append(head_id)
            poss.append([0]+[pos2idx[i] for i in pos]+[0])
            if len(words) > MAX_LEN:
                # 先对句号分段
                word, tag = [], []
                for char, t in zip(words, tags):

                    if char != '。':
                        if char != '\ue236':  # 测试集中有这个字符
                            word.append(char)
                            tag.append(t)
                    else:
                        word.append(char)
                        tag.append(t)
                        sents.append(["[CLS]"] + word[:MAX_LEN] + ["[SEP]"])  # 如果word长度小于maxlen，取的为word的所有内容
                        tags_li.append(['[CLS]'] + tag[:MAX_LEN] + ['[SEP]'])
                        if len(word) > MAX_LEN:
                            sents.append(["[CLS]"] + word[MAX_LEN:] + ["[SEP]"])
                            tags_li.append(['[CLS]'] + tag[MAX_LEN:] + ['[SEP]'])
                        word, tag = [], []
                # 最后的末尾
                if len(word):
                    sents.append(["[CLS]"] + word[:MAX_LEN] + ["[SEP]"])
                    tags_li.append(['[CLS]'] + tag[:MAX_LEN] + ['[SEP]'])
                    word, tag = [], []
            else:
                sents.append(["[CLS]"] + words[:MAX_LEN] + ["[SEP]"])
                tags_li.append(['[CLS]'] + tags[:MAX_LEN] + ['[SEP]'])
        self.sents, self.tags_li, self.heads,self.poss = sents, tags_li, heads,poss

    def __getitem__(self, idx):
        words, tags, heads ,pos= self.sents[idx], self.tags_li[idx], self.heads[idx],self.poss[idx]

        # 将word和tag转成id
        y=-1
        x= []
        entiey_pos1 = []
        entiey_pos2 = []
        pos1=False
        rela_pos = []
        valid_pos = []  # 除了实体位置和cls,sep位置，句子中词的位置全为1

        for w, t in zip(words, tags):
            # tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            tokens = [w]
            xx = tokenizer.convert_tokens_to_ids(tokens)
            # assert len(tokens) == len(xx), f"len(tokens)={len(tokens)}, len(xx)={len(xx)}"

            # 中文没有英文wordpiece后分成几块的情况
            mask = [1] + [0] * (len(tokens) - 1)
            if t in ["[CLS]","[SEP]"]:
                mask=[0]

            if 'B-' in t:
                mask = [0]
                if not pos1:
                    entiey_pos1.append(1)
                    entiey_pos2.append(0)
                    pos1 = True
                else:
                    entiey_pos2.append(1)
                    entiey_pos1.append(0)
            else:
                entiey_pos1.append(0)
                entiey_pos2.append(0)

            if 'R-' in t:
                rela_pos.append(1)
            else:
                rela_pos.append(0)

            if t in rela:  # 修改y
                y = rela.index(t)  # 修改y

            x.extend(xx)
            valid_pos.extend(mask)

        assert len(x)==len(valid_pos) == len(entiey_pos1)== len(entiey_pos2) == len(rela_pos) == \
               len(heads) + 2 == len(pos), f"len(x)={len(x)},  len(masks)={len(valid_pos)}, " \
                        f"len(entiey_pos)={len(entiey_pos1)}, len(rela_pos)={len(rela_pos)}"

        # seqlen
        seqlen = len(x)
        # to string
        words = " ".join(words)
        tags = " ".join(tags)


        # 将head转成adj
        #全图
        adj = np.eye(seqlen)
        for i, headid in enumerate(heads):
            adj[i + 1, headid+1 ] = 1
            adj[headid+1 , i + 1] = 1

        # #最短路径图
        # entity = [i for i, x in enumerate(entiey_pos) if x == 1]
        # edges = []
        # for i, k in enumerate(heads):
        #     edges.append(('{0}'.format(i), '{0}'.format(k)))
        # graph = nx.Graph(edges)
        # shortest_path = nx.shortest_path(graph, source=str(entity[0]), target=str(entity[1]))
        #
        # adj = np.eye(seqlen)
        # for i in range(len(shortest_path) - 1):
        #     adj[int(shortest_path[i]) + 1, int(shortest_path[i + 1]) + 1] = 1
        #     adj[int(shortest_path[i + 1]) + 1, int(shortest_path[i]) + 1] = 1

        # seqlen=list(range(1, seqlen + 1))
        return words, x, valid_pos, tags, y, adj, entiey_pos1,entiey_pos2,rela_pos,pos,seqlen

    def __len__(self):
        return len(self.sents)


def pad(batch):
    '''Pads to the longest sample'''
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)  # 一个batch所有的句子
    tags = f(3)
    y = f(4)

    seqlens = f(-1)
    # maxlen = np.array(seqlens).max()# 选的是一句话里最长的句子长度
    maxlen = 50  #
    word_pos = [[1] * len(sample[1]) + [0] * (maxlen - len(sample[1])) for sample in batch]#用来计算损失，目前效果更好

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]  # 0: <pad>
    x = f(1, maxlen)
    valid_pos =f(2,maxlen)#用来取关系词位置
    entiey_pos1=f(6,maxlen)
    entiey_pos2=f(7,maxlen)
    rela_pos=f(8,maxlen)
    pos=f(9,maxlen)


    adjs = []
    for sample in batch:
        adj1 = sample[5]
        seqlen = sample[-1]
        new_adj = np.eye(maxlen)
        new_adj[:seqlen, :seqlen] = adj1
        adjs.append(new_adj)
    pla=[list(range(1, sample[-1] + 1)) + [0] * (maxlen - sample[-1]) for sample in batch]

    f = torch.LongTensor

    return words, f(x), f(word_pos),f(valid_pos), tags, f(y),f(np.array(adjs)), f(entiey_pos1),f(entiey_pos2),f(rela_pos),f(pos),f(pla)  # X:tokensid,Y:tagsid
#mask :填充的位置为0
##entiey_pos： 用来标记实体的位置
#score： 用来标记关系的位置


