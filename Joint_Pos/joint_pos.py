# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import argparse
from torch.utils import data
# from model import Net
# import calcmetric

from Joint_Pos.net import Bert_GCN_Att
from Joint_Pos.utils import NerDataset, pad
import copy
import random
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")
rela = ['R-execute', "R-work_at", "R-workmate", "R-use", "R-put", "R-involved_person",
     'R-happen_place','R-involved_org','R-happen_time','R-located_in','R-hired_by','R-belong_to']


# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def train(model, iterator, optimizer, criterion, device):
    model.train()
    for i, batch in enumerate(iterator):
        words, x,word_pos,valid_pos,tags,y, adjs, entiey_pos1, entiey_pos2, rela_pos,pos, pla = batch
        x = x.to(device)
        y = y.to(device)
        entiey_pos1 = entiey_pos1.to(device)
        entiey_pos2 = entiey_pos2.to(device)
        rela_pos = rela_pos.to(device)
        adjs = adjs.to(device)
        word_pos = word_pos.to(device)
        valid_pos = valid_pos.to(device)
        pos=pos.to(device)

        entity_att,out = model(x, adjs, entiey_pos1, entiey_pos2,valid_pos,pos)
        optimizer.zero_grad()

        loss = model.loss(entity_att,word_pos,rela_pos,out,y)
        # loss = torch.mean(1 - torch.cosine_similarity(entity_att * word_pos, rela_pos))

        loss.backward()

        optimizer.step()

        # if i == 0:
        #     print("=====sanity check======")
        #     # print("words:", words[0])
        #     print("x:", x.cpu().numpy()[0][:seqlens[0]])
        #     # print("tokens:", tokenizer.convert_ids_to_tokens(x.cpu().numpy()[0])[:seqlens[0]])
        #     print("masks:", word_pos.cpu().numpy()[0][:seqlens[0]])
        #     # print("y:", y.cpu().numpy()[0][:seqlens[0]])
        #     # print("y:", y[0].cpu())
        #     print("tags:", tags[0])
        #     print("entity_pos1:", entiey_pos1.cpu().numpy()[0][:seqlens[0]])
        #     print("entity_pos2:", entiey_pos2.cpu().numpy()[0][:seqlens[0]])
        #     print("score:", rela_pos.cpu().numpy()[0][:seqlens[0]])
        #     print("seqlen:", seqlens[0])
        #     print("=======================")
        #
        # if i % 10 == 0:  # monitoring
        #     print(f"step: {i}, loss: {loss.item()}")
        #     # print(entity_att*word_pos)


def eval(model, iterator, f, device):
    model.eval()

    Words, Valid_pos, Tags, Scores, Seqlens = [], [], [], [], []
    Relas=[]
    Y_hat, Y = [], []
    Entity_att=[]
    OUT=[]
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, word_pos, valid_pos, tags, y, adjs, entiey_pos1, entiey_pos2, rela_pos, pos, pla = batch
            x = x.to(device)
            y = y.to(device)
            entiey_pos1 = entiey_pos1.to(device)
            entiey_pos2 = entiey_pos2.to(device)
            rela_pos = rela_pos.to(device)
            adjs = adjs.to(device)
            word_pos = word_pos.to(device)
            valid_pos = valid_pos.to(device)
            pos = pos.to(device)

            entity_att, out = model(x, adjs, entiey_pos1, entiey_pos2, valid_pos, pos)

            # test_loss = torch.mean(1 - torch.cosine_similarity(entity_att * word_pos, rela_pos))
            test_loss = model.loss(entity_att, word_pos, rela_pos, out, y)
            y_hat = out.argmax(dim=1)

            Words.extend(words)
            Entity_att.extend(entity_att.cpu().numpy().tolist())
            Valid_pos.extend(valid_pos.cpu().numpy().tolist())
            Relas.extend(rela_pos.cpu().numpy().tolist())
            # Seqlens.extend(seqlens.cpu().numpy().tolist())
            Tags.extend(tags)
            Y_hat.extend(y_hat.cpu().numpy().tolist())
            Y.extend(y.cpu().numpy().tolist())
            OUT.extend(out.cpu().numpy().tolist())

            # if i % 10 == 0:  # monitoring
            #     print(f"step: {i}, loss: {test_loss.item()}")
            # # print(entity_att[0])


    ## gets results and save

    pos_count = 0
    class_count=0
    Pred = []
    for y_hat, y, rela_pos, entity_att,valid_pos in zip(Y_hat, Y, Relas, Entity_att,Valid_pos):
    # for  rela_pos, entity_att,valid_pos in zip( Relas, Entity_att,Valid_pos):

        #计算分类正确率
        if y_hat==y:
            class_count+=1

        #计算位置正确率
        entity_att = (np.array(entity_att)*np.array(valid_pos)).tolist()

        out_copy = copy.deepcopy(entity_att)
        # 正确位置是否在前三关系分数
        pre_index = []
        # 得到预测的前三关系位置
        for _ in range(3):
            number = max(out_copy)
            index = out_copy.index(number)
            out_copy[index] = 0
            pre_index.append(index)


        true_index = [i for i, x in enumerate(rela_pos) if x == 1]
        # print(pre_index,true_index)
        if pre_index==[1,1,1]:print(entity_att)
        if len(set(true_index) & set(pre_index)) >= 1:
            pos_count += 1  # 统计正确的条目

        pre_index = [str(i) for i in pre_index]
        Pred.append(' '.join(pre_index)+'\n')

        # for w, t, s,e,o in zip(words.split()[1:-1], tags.split()[1:-1],score[1:seqlen-1],entiey_pos[1:seqlen-1] ,out):
        #     lines.append(f"{w} {t} {s} {e} {o}\n")
        # lines.append("\n")

    count=len(Y)
    pos_acc = pos_count / count
    class_acc=class_count/count
    print(f'pos_acc: {pos_count}/{count}={pos_acc}')
    print(f'class_acc: {class_count}/{count}={class_acc}')
    res = classification_report(Y, Y_hat,digits=5)
    print(res)
    # print('accuracy: ',   res['accuracy'])
    # print('macro avg-f1-score:', res['macro avg']['f1-score'])
    # print('weighted avg-f1-score:', res['weighted avg']['f1-score'])

    final = f + '-' + str(class_acc)+'.csv'
    # with open(final, 'w', encoding='utf-8') as fw:
    #     fw.writelines(Pred)

    np.savetxt(final, np.array(OUT), delimiter=",")
    print(final)
    # np.savetxt('Y.csv', np.array(Y), delimiter=",")



    return pos_acc,class_acc



#D:\Anaconda\envs\pytorch\python.exe
if __name__ == "__main__":
    all_result=[]
    # for i in np.arange(0.0002,0.001,0.0001):
    #     for j in np.arange(0.2,0.5,0.1):
    #         for k in np.arange(0.1,0.6,0.1):

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0004)
    parser.add_argument("--gcn_drop", type=float, default=0.5)
    parser.add_argument("--class_drop", type=float, default=0.2)
    parser.add_argument("--n_epochs", type=int, default=30)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--top_rnns", dest="top_rnns", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/06")
    parser.add_argument("--trainset", type=str,
                        default="C:/Users/1/Desktop/实验/bilstm-att/processed/2023-11-26train50.txt")
    parser.add_argument("--validset", type=str,
                        default="C:/Users/1/Desktop/实验/bilstm-att/processed/2023-11-26test50.txt")
    hp = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def setup_seed(seed):
        torch.manual_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)


    # 设置随机数种子
    setup_seed(1)

    model = Bert_GCN_Att(rela, hp.gcn_drop, hp.class_drop).cuda()
    print('Initial model Done')
    # model = nn.DataParallel(model)

    train_dataset = NerDataset(hp.trainset)
    eval_dataset = NerDataset(hp.validset)
    print('Load Data Done')

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 collate_fn=pad)  # pad为utils里的pad函数
    eval_iter = data.DataLoader(dataset=eval_dataset,
                                batch_size=hp.batch_size,
                                shuffle=False,
                                num_workers=0,
                                collate_fn=pad)

    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print('Start Train...,')
    result = [0.0, 0.0]
    for epoch in range(1, hp.n_epochs + 1):  # 每个epoch对dev集进行测试

        print(f"=========train at epoch={epoch}=========")
        train(model, train_iter, optimizer, criterion, device)

        if epoch > 0:

            print(f"=========eval at epoch={epoch}=========")
            if not os.path.exists(hp.logdir):
                os.makedirs(hp.logdir)
            fname = os.path.join(hp.logdir, str(epoch))
            pos_acc, class_acc = eval(model, eval_iter, fname, device)
            if class_acc > result[1]:
                result[0] = pos_acc
                result[1] = class_acc
            if class_acc == result[1] and pos_acc > result[0]:
                result[0] = pos_acc
    # result.extend([i, j, k])
    # all_result.append(result)
    # print('best result:', [i, j, k])
    print(f'pos_acc: {result[0] * 500}/500={result[0]}')
    print(f'class_acc: {result[1] * 500}/500={result[1]}')




