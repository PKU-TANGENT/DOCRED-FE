import json
import math
import os
import pickle
import random
from collections import defaultdict

import dgl
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
from transformers import *

from models.GAIN import Bert
from utils import get_cuda

IGNORE_INDEX = -100

# 继承data 中的IterableDataset 
class DGLREDataset(IterableDataset):

    def __init__(self, src_file, save_file, word2id, ner2id, rel2id,
                 dataset_type='train', instance_in_train=None, opt=None):

        super(DGLREDataset, self).__init__()

        # record training set mention triples
        self.instance_in_train = set([]) if instance_in_train is None else instance_in_train
        self.data = None
        self.document_max_length = 512
        self.INTRA_EDGE = 0
        self.INTER_EDGE = 1
        self.LOOP_EDGE = 2
        self.count = 0

        print('Reading data from {}.'.format(src_file))
        if os.path.exists(save_file):
            with open(file=save_file, mode='rb') as fr:
                info = pickle.load(fr)
                self.data = info['data']
                self.instance_in_train = info['intrain_set']
            print('load preprocessed data from {}.'.format(save_file))

        else:
            with open(file=src_file, mode='r', encoding='utf-8') as fr:
                ori_data = json.load(fr)
            print('loading..')
            self.data = []

            for i, doc in enumerate(ori_data):

                title, entity_list, labels, sentences = \
                    doc['title'], doc['vertexSet'], doc.get('labels', []), doc['sents']

                Ls = [0]
                L = 0
                for x in sentences:
                    L += len(x)
                    Ls.append(L)
                for j in range(len(entity_list)):
                    for k in range(len(entity_list[j])):
                        sent_id = int(entity_list[j][k]['sent_id'])
                        entity_list[j][k]['sent_id'] = sent_id

                        dl = Ls[sent_id]
                        pos0, pos1 = entity_list[j][k]['pos']
                        entity_list[j][k]['global_pos'] = (pos0 + dl, pos1 + dl)

                # generate positive examples
                train_triple = []
                new_labels = []
                for label in labels:
                    head, tail, relation, evidence = label['h'], label['t'], label['r'], label['evidence']
                    assert (relation in rel2id), 'no such relation {} in rel2id'.format(relation)
                    label['r'] = rel2id[relation]

                    train_triple.append((head, tail))

                    label['in_train'] = False

                    # record training set mention triples and mark it for dev and test set
                    for n1 in entity_list[head]:
                        for n2 in entity_list[tail]:
                            mention_triple = (n1['name'], n2['name'], relation)
                            if dataset_type == 'train':
                                self.instance_in_train.add(mention_triple)
                            else:
                                if mention_triple in self.instance_in_train:
                                    label['in_train'] = True
                                    break

                    new_labels.append(label)

                # generate negative examples
                na_triple = []
                for j in range(len(entity_list)):
                    for k in range(len(entity_list)):
                        if j != k and (j, k) not in train_triple:
                            na_triple.append((j, k))

                # generate document ids
                words = []
                for sentence in sentences:
                    for word in sentence:
                        words.append(word)
                if len(words) > self.document_max_length:
                    words = words[:self.document_max_length]

                word_id = np.zeros((self.document_max_length,), dtype=np.int32)
                pos_id = np.zeros((self.document_max_length,), dtype=np.int32)
                ner_id = np.zeros((self.document_max_length,), dtype=np.int32)
                mention_id = np.zeros((self.document_max_length,), dtype=np.int32)

                for iii, w in enumerate(words):
                    word = word2id.get(w.lower(), word2id['UNK'])
                    word_id[iii] = word

                entity2mention = defaultdict(list)
                mention_idx = 1
                already_exist = set()  # dealing with NER overlapping problem
                for idx, vertex in enumerate(entity_list, 1):
                    for v in vertex:
                        sent_id, (pos0, pos1), ner_type = v['sent_id'], v['global_pos'], v['type']
                        if (pos0, pos1) in already_exist:
                            continue
                        pos_id[pos0:pos1] = idx
                        ner_id[pos0:pos1] = ner2id[ner_type]
                        mention_id[pos0:pos1] = mention_idx
                        entity2mention[idx].append(mention_idx)
                        mention_idx += 1
                        already_exist.add((pos0, pos1))

                # construct graph
                graph = self.create_graph(Ls, mention_id, pos_id, entity2mention)

                # construct entity graph & path
                entity_graph, path = self.create_entity_graph(Ls, pos_id, entity2mention)

                assert pos_id.max() == len(entity_list)
                assert mention_id.max() == graph.number_of_nodes() - 1

                overlap = doc.get('overlap_entity_pair', [])
                new_overlap = [tuple(item) for item in overlap]

                self.data.append({
                    'title': title,
                    'entities': entity_list,
                    'labels': new_labels,
                    'na_triple': na_triple,
                    'word_id': word_id,
                    'pos_id': pos_id,
                    'ner_id': ner_id,
                    'mention_id': mention_id,
                    'entity2mention': entity2mention,
                    'graph': graph,
                    'entity_graph': entity_graph,
                    'path': path,
                    'overlap': new_overlap
                })

            # save data
            with open(file=save_file, mode='wb') as fw:
                pickle.dump({'data': self.data, 'intrain_set': self.instance_in_train}, fw)
            print('finish reading {} and save preprocessed data to {}.'.format(src_file, save_file))

        if opt.k_fold != "none":
            k_fold = opt.k_fold.split(',')
            k, total = float(k_fold[0]), float(k_fold[1])
            a = (k - 1) / total * len(self.data)
            b = k / total * len(self.data)
            self.data = self.data[:a] + self.data[b:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)

    def create_graph(self, Ls, mention_id, entity_id, entity2mention):

        d = defaultdict(list)

        # add intra-entity edges
        for _, mentions in entity2mention.items():
            for i in range(len(mentions)):
                for j in range(i + 1, len(mentions)):
                    d[('node', 'intra', 'node')].append((mentions[i], mentions[j]))
                    d[('node', 'intra', 'node')].append((mentions[j], mentions[i]))

        if d[('node', 'intra', 'node')] == []:
            d[('node', 'intra', 'node')].append((entity2mention[1][0], 0))

        for i in range(1, len(Ls)):
            tmp = dict()
            for j in range(Ls[i - 1], Ls[i]):
                if mention_id[j] != 0:
                    tmp[mention_id[j]] = entity_id[j]
            mention_entity_info = [(k, v) for k, v in tmp.items()]

            # add self-loop & to-globle-node edges
            for m in range(len(mention_entity_info)):
                # self-loop
                # d[('node', 'loop', 'node')].append((mention_entity_info[m][0], mention_entity_info[m][0]))

                # to global node
                d[('node', 'global', 'node')].append((mention_entity_info[m][0], 0))
                d[('node', 'global', 'node')].append((0, mention_entity_info[m][0]))

            # add inter edges
            for m in range(len(mention_entity_info)):
                for n in range(m + 1, len(mention_entity_info)):
                    if mention_entity_info[m][1] != mention_entity_info[n][1]:
                        # inter edge
                        d[('node', 'inter', 'node')].append((mention_entity_info[m][0], mention_entity_info[n][0]))
                        d[('node', 'inter', 'node')].append((mention_entity_info[n][0], mention_entity_info[m][0]))

        # add self-loop for global node
        # d[('node', 'loop', 'node')].append((0, 0))
        if d[('node', 'inter', 'node')] == []:
            d[('node', 'inter', 'node')].append((entity2mention[1][0], 0))

        graph = dgl.heterograph(d)

        return graph

    def create_entity_graph(self, Ls, entity_id, entity2mention):

        graph = dgl.DGLGraph()
        graph.add_nodes(entity_id.max())

        d = defaultdict(set)

        for i in range(1, len(Ls)):
            tmp = set()
            for j in range(Ls[i - 1], Ls[i]):
                if entity_id[j] != 0:
                    tmp.add(entity_id[j])
            tmp = list(tmp)
            for ii in range(len(tmp)):
                for jj in range(ii + 1, len(tmp)):
                    d[tmp[ii] - 1].add(tmp[jj] - 1)
                    d[tmp[jj] - 1].add(tmp[ii] - 1)
        a = []
        b = []
        for k, v in d.items():
            for vv in v:
                a.append(k)
                b.append(vv)
        graph.add_edges(a, b)

        path = dict()
        for i in range(0, graph.number_of_nodes()):
            for j in range(i + 1, graph.number_of_nodes()):
                a = set(graph.successors(i).numpy())
                b = set(graph.successors(j).numpy())
                c = [val + 1 for val in list(a & b)]
                path[(i + 1, j + 1)] = c

        return graph, path


class BERTDGLREDataset(IterableDataset):

    def __init__(self, src_file, save_file, word2id, ner2id, rel2id,
                 dataset_type='train', instance_in_train=None, opt=None):

        super(BERTDGLREDataset, self).__init__()
        # record training set mention triples 
        # set([]) 和 set() 是同样的作用
        self.instance_in_train = set([]) if instance_in_train is None else instance_in_train
        self.data = None
        self.document_max_length = 512
        
        # 如下这几个参数的含义与作用？
        self.INFRA_EDGE = 0
        self.INTER_EDGE = 1
        self.LOOP_EDGE = 2
        self.count = 0

        print('Reading data from {}.'.format(src_file))
        if os.path.exists(save_file):
            with open(file=save_file, mode='rb') as fr:
                info = pickle.load(fr) # info <class 'dict'> key-value 是：data, intrain_set
                # list, 里面是dict
                
                """                
                {
                    title:,
                    entities:"",
                    labels:"",
                    na_triple:"",
                    word_id:"",
                    pos_id:"",
                    ner_id:"",
                    mention_id:"",
                    entity2mention:,
                    graph:,
                    entity_graph:,
                    path:,
                    overlap:[]
                }
                """
                self.data = info['data']
                # ('Justin Broadrick', 'Earache Records', 'P264')
                self.instance_in_train = info['intrain_set'] # 是一个三元组组成的list
            print('load preprocessed data from {}.'.format(save_file))
            
        else: # 在没有缓存数据时（.pkl文件不存在）进行的操作。 数据处理操作需要认真阅读！
            bert = Bert(BertModel, 'bert-base-uncased', opt.bert_path)

            with open(file=src_file, mode='r', encoding='utf-8') as fr:
                ori_data = json.load(fr)
            print('loading..')
            self.data = []

            pbar = tqdm(range(len(ori_data)))
            for i, doc in enumerate(ori_data):
                # doc 中的内容只有如下四项，分别是 title, vertexSet, labels, sents
                title, entity_list, labels, sentences = \
                    doc['title'], doc['vertexSet'], doc.get('labels', []), doc['sents'] 
                Ls = [0] # Ls[i] 表示的就是第i条句子开始的绝对长度。直接以原始setence中的word为基准
                L = 0
                # step1. 遍历每个的 sentence 的长度，累计得到当前的总长度L， 并将其放到 Ls中。
                for x in sentences: 
                    L += len(x)
                    Ls.append(L)
                
                entity_types = []
                # step2. 遍历每个entity
                for j in range(len(entity_list)): 
                    for k in range(len(entity_list[j])): # 找出当前entity下的mention个数
                                                
                        sent_id = int(entity_list[j][k]['sent_id']) # 找出当前这个mention 所在的sent_id ，但是如果一个mention
                        # 下面这行代码应该无用
                        # entity_list[j][k]['sent_id'] = sent_id # 即使相同的mention，但是可能出现在句中的位置不同，所以这里放的是[实体_id][mention_id]['send_id'] = sent_id

                        dl = Ls[sent_id]  # dl 是对应第 sent_id 个句子的长度
                        mention_start, mention_end = entity_list[j][k]['pos'] # 得到这个mention 的 [start,end] 坐标
                        entity_list[j][k]['global_pos'] = (mention_start + dl, mention_end + dl) # global_pos 计算的就是全局位置
                    entity_types.append(entity_list[j][0]['type'])
                
                # step3. 遍历labels
                # generate positive examples
                train_triple = []
                new_labels = []  # 产生新的label标签
                for label in labels:
                    head, tail, relation, evidence = label['h'], label['t'], label['r'], label['evidence'] # 从原始数据中取出对应的字段
                    assert (relation in rel2id), 'no such relation {} in rel2id'.format(relation) # 判断是否有对应的relation
                    label['r'] = rel2id[relation] # 将rel => id
                    train_triple.append((head, tail)) # 记录正样本数据
                    label['in_train'] = False  # 为什么这里初始化为False?
                    
                    # record training set mention triples and mark it for dev and test set
                    for n1 in entity_list[head]:
                        for n2 in entity_list[tail]:
                            mention_triple = (n1['name'], n2['name'], relation)  # 这相当于根据所有的mention都生成一个train example  => 这会生成很多个训练数据
                            if dataset_type == 'train':
                                self.instance_in_train.add(mention_triple)
                            else: # 因为这里存在既加载train，又加载dev/test。所以为了判断是否train中的标签会出现dev中，所以就用来这个标记。
                                if mention_triple in self.instance_in_train:
                                    label['in_train'] = True
                                    break
                                       
                    new_labels.append(label)

                # generate negative examples  => 其实负样本的选择也是很重要的，所以这里也是一个课题可以作为改进
                na_triple = []
                for j in range(len(entity_list)):
                    for k in range(len(entity_list)):
                        if j != k and (j, k) not in train_triple:  # 如果不在正样本数据中，则作为负样本使用
                            na_triple.append((j, k))

                # generate document ids
                words = [] # 取出sentence 中的所有单词
                # a = []
                for sentence in sentences:
                    words.extend(sentence)
                    # for word in sentence:
                    #     a.append(word)
                # print(a)
                # print(words)
                # bert_token, bert_starts, bert_subwords 用bert_xx 是为了表示要送给bert处理
                # bert_starts[i] 表示第i个word 在bert_token 中的起始下标。因为有拆分情况，所以需要记录一下
                bert_token, bert_starts, bert_subwords = bert.subword_tokenize_to_ids(words)
                # zeros() 函数会传入一个shape，即下面的(self.document_max_length,) 这里将传递给杜设置成document_max_length，就可以猜测是要对整个tokenizer之后的序列进行一个标注，否则没有必要搞这么个长度
                word_id = np.zeros((self.document_max_length,), dtype=np.int32) 
                pos_id = np.zeros((self.document_max_length,), dtype=np.int32)  # 这个改做pos2entityid 比较合理
                ner_id = np.zeros((self.document_max_length,), dtype=np.int32) # 默认的初始值就代表这个不是NER
                
                # 这个改做叫pos2mention_id 更好
                mention_id = np.zeros((self.document_max_length,), dtype=np.int32)
                word_id[:] = bert_token[0] # 其实就是 tokenizer处理后的 input_ids

                entity2mention = defaultdict(list) # 记住每个entity 对应mention的id
                mention_idx = 1 # 对mention的下标计数，注意是从1开始。 即统计一篇 doc 中的所有mention个数，所以是个全局变量
                already_exist = set() # 记录两个
                for idx, vertex in enumerate(entity_list, 1): # 对所有的entity遍历
                    for v in vertex: # vertex 代表的是这个entity的所有 mention 集合
                        sent_id, (pos0, pos1), ner_type = v['sent_id'], v['global_pos'], v['type']  # 注意这里取得是绝对位置
                        pos0 = bert_starts[pos0] # 返回在tokenizer之后（原words在）位置pos0处的位置
                        pos1 = bert_starts[pos1] if pos1 < len(bert_starts) else 1024 # TODO 这个值为啥选择1024？
                        # 下面这两种if会存在吗？
                        if (pos0, pos1) in already_exist:
                            continue

                        if pos0 >= len(pos_id):
                            continue
                        
                        pos_id[pos0:pos1] = idx # 表示pos0 -> pos1 这个位置是第idx个实体（用的是idx下标）
                        ner_id[pos0:pos1] = ner2id[ner_type]  # 标记整个位置的 ner_type 情况，标记成id。 ner_id[i] 表示成第i个位置是个ner，且这个ner的类型对应的id是 ner2id[ner_type]
                        mention_id[pos0:pos1] = mention_idx # 记录从pos:pos1 这个位置是一个mention（用的是mention_idx下标），对应第mention_idx个mention
                        entity2mention[idx].append(mention_idx)  
                        mention_idx += 1  
                        already_exist.add((pos0, pos1))
                replace_i = 0
                if entity2mention[idx] == []:
                    idx = len(entity_list) # 这行代码放在这个位置，应该才ok 
                    entity2mention[idx].append(mention_idx)
                    while mention_id[replace_i] != 0:
                        replace_i += 1
                    mention_id[replace_i] = mention_idx
                    pos_id[replace_i] = idx
                    ner_id[replace_i] = ner2id[vertex[0]['type']]
                    mention_idx += 1

                new_Ls = [0] # 之前的Ls是按照原始sentence的分词搞的，接下来就要用基于bert tokenizer 之后的来搞
                for ii in range(1, len(Ls)):
                    # bert_starts[Ls[ii]] 找出对应 Ls[ii] 这个word 得到的token下标
                    new_Ls.append(bert_starts[Ls[ii]] if Ls[ii] < len(bert_starts) else len(bert_subwords))
                Ls = new_Ls

                # construct graph
                mention_graph = self.create_mention_graph(Ls, mention_id, pos_id, entity2mention)

                # construct entity graph & path
                entity_graph, path = self.create_entity_graph(Ls, pos_id, entity2mention)

                assert pos_id.max() == len(entity_list)
                assert mention_id.max() == mention_graph.number_of_nodes() - 1
                # 没有明白这个 overlap 的作用是什么？
                overlap = doc.get('overlap_entity_pair', [])
                new_overlap = [tuple(item) for item in overlap]

                self.data.append(
                    {
                    'title': title,
                    'entities': entity_list,
                    'entity_types': entity_types,
                    'labels': new_labels,
                    'na_triple': na_triple,
                    'word_id': word_id,
                    'pos_id': pos_id,
                    'ner_id': ner_id,
                    'mention_id': mention_id,
                    'entity2mention': entity2mention,
                    'graph': mention_graph,
                    'entity_graph': entity_graph,
                    'path': path,
                    'overlap': new_overlap
                    }
                )
                pbar.update(1)

            # save data
            with open(file=save_file, mode='wb') as fw:
                pickle.dump({'data': self.data, 'intrain_set': self.instance_in_train}, fw)
            print('finish reading {} and save preprocessed data to {}.'.format(src_file, save_file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)

    # 这创建的是 mention graph
    def create_mention_graph(self, Ls, mention_id, entity_id, entity2mention):
        data_dict = defaultdict(list) # 新建一个存在默认值的dict，这个值最后就是用于创建图的
        # add intra edges
        for _, mentions in entity2mention.items():
            for i in range(len(mentions)): # 构建相同 mention 之间的图
                for j in range(i + 1, len(mentions)):                    
                    data_dict[('node', 'intra', 'node')].append((mentions[i], mentions[j])) # 用的是同一个 mentions，所以代表的是intra edges。 mentions[i] 代表第i个mention。这就说明第i个mention 和 第j个mention之间有边。
                    data_dict[('node', 'intra', 'node')].append((mentions[j], mentions[i])) # 相反，第j个mention和第i个mention之间也有边
        # 特殊情况判断
        if data_dict[('node', 'intra', 'node')] == []: # 如果为空集
            data_dict[('node', 'intra', 'node')].append((entity2mention[1][0], 0))

        # 下面这个获取 mention_entity_info 的步骤还需要再研究一下
        for i in range(1, len(Ls)): # 从1开始是因为前面有 [CLS]。 这遍历的结果相当于从第i个sentence对应的token之后的起始下标
            tmp = dict()
            for j in range(Ls[i - 1], Ls[i]): # 第i个sentence 的起始下标和终止下标
                if mention_id[j] != 0: # 如果当前这个位置的token 是mention的一部分
                    # print(mention_id[j],"=>",entity_id[j])
                    tmp[mention_id[j]] = entity_id[j]  # entity_id[j] 表示的是j这个位置对应的entity下标。这就相当于找出 mention => entity 之间的对应关系，即第mention_id[j] 个mention对应 第entity_id[j] 个entity
            mention_entity_info = [(k, v) for k, v in tmp.items()] # TODO mention => entity。 这么写是不是有点儿复杂了？ 如果只想找出mention 序号到 entity 的关系， 应该不需要这么复杂的判断啊 

            # add self-loop & to-globle-node edges
            for m in range(len(mention_entity_info)):
                # self-loop
                # d[('node', 'loop', 'node')].append((mention_entity_info[m][0], mention_entity_info[m][0]))

                # to global node
                # 这里应该就是添加 mention => Document Node 的边
                data_dict[('node', 'global', 'node')].append((mention_entity_info[m][0], 0))
                data_dict[('node', 'global', 'node')].append((0, mention_entity_info[m][0]))

            # add inter edges
            for m in range(len(mention_entity_info)):
                for n in range(m + 1, len(mention_entity_info)):
                    if mention_entity_info[m][1] != mention_entity_info[n][1]: # [m][1], [n][1] 都是取第二个值，就是关注 entity
                        # inter edge  => 就是 Inter-Entity Edge  指的就是同一实体不同mention之间需要相互连接                    
                        data_dict[('node', 'inter', 'node')].append((mention_entity_info[m][0], mention_entity_info[n][0]))
                        data_dict[('node', 'inter', 'node')].append((mention_entity_info[n][0], mention_entity_info[m][0]))

        # add self-loop for global node
        # d[('node', 'loop', 'node')].append((0, 0))
        if data_dict[('node', 'inter', 'node')] == []:
            data_dict[('node', 'inter', 'node')].append((entity2mention[1][0], 0))

        graph = dgl.heterograph(data_dict)
        return graph


    # Entity level 的图之间的边是怎么构建的？ =>  合并所有实体间的边，这些边连接相同的两个mention
    # TODO 这里的建边过程是否还可以值得优化一下？
    def create_entity_graph(self, Ls, entity_id, entity2mention):
        # 新建一个空图
        graph = dgl.DGLGraph()
        # graph 中的节点编号是 [0,entity_id.max()-1]
        graph.add_nodes(entity_id.max()) # 有多少个实体，节点就是多少。 先把节点数表示出来，但是这些节点的具体特征留到后面再做处理
        d = defaultdict(set)

        for i in range(1, len(Ls)):
            tmp = set() # 判断当前 sentence 中有几个实体，并使用set记录它们的id
            for j in range(Ls[i - 1], Ls[i]):
                if entity_id[j] != 0:
                    tmp.add(entity_id[j])
            tmp = list(tmp)  
            for ii in range(len(tmp)): # 这两个for循环是用来干什么的？=> 在同一个sentence之间的实体间建立关系（边）
                for jj in range(ii + 1, len(tmp)):
                    d[tmp[ii] - 1].add(tmp[jj] - 1) # 建双向边
                    d[tmp[jj] - 1].add(tmp[ii] - 1)
        
        # 将d拆分，形成一一对应的数组，然后交由图进行创建边的操作
        a = [] 
        b = []
        for k, v in d.items():
            for vv in v:
                a.append(k)
                b.append(vv)
        graph.add_edges(a, b) # 为图添加边的信息
        
        # 定义一个path，用于reasoning mechanism。 下面这个代码的逻辑就是：找出(i+1,j+1)的中间节点有哪些（集合c）。
        path = dict()
        for i in range(0, graph.number_of_nodes()):
            for j in range(i + 1, graph.number_of_nodes()):
                a = set(graph.successors(i).numpy()) # 找出节点i的子节点。即节点i指向的节点
                b = set(graph.successors(j).numpy()) 
                c = [val + 1 for val in list(a & b)] # a&b 求出集合的交集
                path[(i + 1, j + 1)] = c # TODO 这里为什么i+1？ # 代表的含义：(i+1,j+1):c 表示节点i和节点j都可以到达的节点集合为c。 又因为这里的图是无向图（两个单向）也就是说可以通过集合c，使得i,j互联
                # 不清楚为什么这里要搞成 (i+1,j+1) 的形式，难道直接使用 (i,j) 不行吗？我觉得是可以的

        return graph, path


class DGLREDataloader(DataLoader):
    # 使用num_workers 加速数据的准备过程
    # 这个 h_t_limit_per_batch 是什么意思？
    # TODO h_t_limit 又是啥？=> 我猜测是 h_entity 和 tail_entity 的连接数。这里的1722 是因为 42*42-42 = 1722
    def __init__(self, dataset, batch_size, shuffle=False, h_t_limit_per_batch=300, h_t_limit=1722, relation_num=97,max_length=512, negativa_alpha=0.0, dataset_type='train'):
        super(DGLREDataloader, self).__init__(dataset, batch_size=batch_size) # 初始化 DataLoader 用多线程
        self.shuffle = shuffle
        self.length = len(self.dataset)
        self.max_length = max_length
        self.negativa_alpha = negativa_alpha
        self.dataset_type = dataset_type

        self.h_t_limit_per_batch = h_t_limit_per_batch
        self.h_t_limit = h_t_limit
        self.relation_num = relation_num
        self.dis2idx = np.zeros((512), dtype='int64')
        self.dis2idx[1] = 1
        self.dis2idx[2:] = 2
        self.dis2idx[4:] = 3
        self.dis2idx[8:] = 4
        self.dis2idx[16:] = 5
        self.dis2idx[32:] = 6
        self.dis2idx[64:] = 7
        self.dis2idx[128:] = 8
        self.dis2idx[256:] = 9
        self.dis_size = 20

        self.order = list(range(self.length))        

    def __iter__(self):
        # shuffle
        if self.shuffle:
            random.shuffle(self.order)
            self.data = [self.dataset[idx] for idx in self.order]
        else:
            self.data = self.dataset
        batch_num = math.ceil(self.length / self.batch_size)  # 计算按照当前的batch_size设置，会有多少个batch？
        # 根据当前的idx，返回每个batch需要的数据
        self.batches = [self.data[idx * self.batch_size: min(self.length, (idx + 1) * self.batch_size)]
                        for idx in range(0, batch_num)]
        self.batches_order = [self.order[idx * self.batch_size: min(self.length, (idx + 1) * self.batch_size)]
                              for idx in range(0, batch_num)] # 取出batch对应 的 order
        
        # 查看每个小 batch 里面
        for idx, minibatch in enumerate(self.batches):
            cur_bsz = len(minibatch) # 因为每个batch 不一定是整除的，所以这里先计算一下当前batch 中的个数
            # 这个操作？？ => 让tensor 变成0 tensor。 那么下面这个操作是否是多余的？我直接在上面生成的时候直接使用zeros()怎么样？
            # for mapping in [context_word_ids, context_pos_ids, context_ner_ids, context_mention_ids,
            #                 context_word_mask, context_word_length,
            #                 ht_pairs, ht_pair_distance, relation_multi_label, relation_mask, relation_label]:
            #     if mapping is not None:
            #         mapping.zero_()
            # 原代码是将如下声明 tensor 的代码放到for循环的外侧的，我将其放到内测，同时取消上面这个for 循环
            context_word_ids = torch.zeros(self.batch_size, self.max_length,dtype=torch.long).cpu()
            context_pos_ids = torch.zeros(self.batch_size, self.max_length,dtype=torch.long).cpu()
            context_ner_ids = torch.zeros(self.batch_size, self.max_length,dtype=torch.long).cpu()
            context_mention_ids = torch.zeros(self.batch_size, self.max_length,dtype=torch.long).cpu()
            context_word_mask = torch.zeros(self.batch_size, self.max_length,dtype=torch.long).cpu()
            context_word_length = torch.zeros(self.batch_size,dtype=torch.long).cpu()
            ht_pairs = torch.zeros(cur_bsz, self.h_t_limit, 2,dtype=torch.long).cpu()
            ht_pair_distance = torch.zeros(self.batch_size, self.h_t_limit,dtype=torch.long).cpu()
            relation_multi_label = torch.zeros(self.batch_size, self.h_t_limit, self.relation_num).cpu() # zeros 得到的结果是float
            relation_label = torch.zeros(self.batch_size, self.h_t_limit,dtype=torch.long).cpu()
            relation_mask = torch.zeros(self.batch_size, self.h_t_limit,dtype=torch.long).cpu() # 不理解这里的realtion_mask 的作用
            relation_label.fill_(IGNORE_INDEX)

            max_h_t_cnt = 0

            label_list = []
            L_vertex = []
            titles = []
            entity_types = []
            indexes = [] # TODO ?
            graph_list = []
            entity_graph_list = []
            entity2mention_table = [] # 这个是什么？ => 每篇 doc 形成的一个矩阵（entity 和 mention的对应关系）叫做entity2mention。将batch下的每篇doc得到的矩阵放到同一个list中就是
            path_table = []
            overlaps = []
            # 对这个batch 中的数据进行处理
            for i, example in enumerate(minibatch):
                title, entities, labels, na_triple, word_id, pos_id, ner_id, mention_id, entity2mention, graph, entity_graph, path = \
                    example['title'], example['entities'], example['labels'], example['na_triple'], \
                    example['word_id'], example['pos_id'], example['ner_id'], example['mention_id'], example[
                        'entity2mention'], example['graph'], example['entity_graph'], example['path']
                entity_type = example['entity_types']
                graph_list.append(graph) # 整个batch 的图得放到一个list 中
                entity_graph_list.append(entity_graph)
                path_table.append(path)
                overlaps.append(example['overlap'])
                # entity2mention_t 是个矩阵的形式，为啥要多加1?
                entity2mention_t = get_cuda(torch.zeros((pos_id.max() + 1, mention_id.max() + 1)))
                for e, ms in entity2mention.items(): # entity2mention 是个dict。 这两个for循环是为了构建一个邻接矩阵，将entity2mention 转换成一个邻接矩阵
                    for m in ms:
                        entity2mention_t[e, m] = 1
                entity2mention_table.append(entity2mention_t)

                L = len(entities)  # 当前这篇doc 中entity 的数量
                word_num = word_id.shape[0]
                # 将word_id 的值放到context_word_id[i,:word_num] 中。 这里的 :word_num 是有点儿多余
                context_word_ids[i, :word_num].copy_(torch.from_numpy(word_id))
                context_pos_ids[i, :word_num].copy_(torch.from_numpy(pos_id))
                context_ner_ids[i, :word_num].copy_(torch.from_numpy(ner_id))
                context_mention_ids[i, :word_num].copy_(torch.from_numpy(mention_id))

                idx2label = defaultdict(list)
                label_set = {}
                for label in labels:
                    head, tail, relation, intrain, evidence = \
                        label['h'], label['t'], label['r'], label['in_train'], label['evidence']
                    idx2label[(head, tail)].append(relation)
                    label_set[(head, tail, relation)] = intrain # 这个intrain 是判断当前这条样例是否出现在train中的标志

                label_list.append(label_set)

                if self.dataset_type == 'train':
                    train_tripe = list(idx2label.keys())
                    for j, (h_idx, t_idx) in enumerate(train_tripe):
                        hlist, tlist = entities[h_idx], entities[t_idx]
                        ht_pairs[i, j, :] = torch.Tensor([h_idx + 1, t_idx + 1])
                        label = idx2label[(h_idx, t_idx)]

                        delta_dis = hlist[0]['global_pos'][0] - tlist[0]['global_pos'][0]
                        if delta_dis < 0:
                            ht_pair_distance[i, j] = -int(self.dis2idx[-delta_dis]) + self.dis_size // 2
                        else:
                            ht_pair_distance[i, j] = int(self.dis2idx[delta_dis]) + self.dis_size // 2

                        for r in label:
                            relation_multi_label[i, j, r] = 1

                        relation_mask[i, j] = 1
                        rt = np.random.randint(len(label))
                        relation_label[i, j] = label[rt]

                    lower_bound = len(na_triple)
                    if self.negativa_alpha > 0.0:
                        random.shuffle(na_triple)
                        lower_bound = int(max(20, len(train_tripe) * self.negativa_alpha))

                    for j, (h_idx, t_idx) in enumerate(na_triple[:lower_bound], len(train_tripe)):
                        hlist, tlist = entities[h_idx], entities[t_idx]
                        ht_pairs[i, j, :] = torch.Tensor([h_idx + 1, t_idx + 1])

                        delta_dis = hlist[0]['global_pos'][0] - tlist[0]['global_pos'][0]
                        if delta_dis < 0:
                            ht_pair_distance[i, j] = -int(self.dis2idx[-delta_dis]) + self.dis_size // 2
                        else:
                            ht_pair_distance[i, j] = int(self.dis2idx[delta_dis]) + self.dis_size // 2

                        relation_multi_label[i, j, 0] = 1
                        relation_label[i, j] = 0
                        relation_mask[i, j] = 1

                        max_h_t_cnt = max(max_h_t_cnt, len(train_tripe) + lower_bound)
                else:
                    j = 0 
                    for h_idx in range(L): # 因为要判断所有的entity，所以用的是双重for循环
                        for t_idx in range(L):
                            if h_idx != t_idx: # 这里判断是否赋值 ht_pairs 的条件只是 h_idx 和 t_idx 的值的比较
                                hlist, tlist = entities[h_idx], entities[t_idx] # 分别获取每个实体下的所有mention 信息
                                # TODO ht_pairs 是什么意思？  +1 是因为？ => 难道就只是想看h_entity 和 tail_entity 之间能否组成一对？所以就叫ht_pair?
                                ht_pairs[i, j, :] = torch.Tensor([h_idx + 1, t_idx + 1]) 
                                relation_mask[i, j] = 1
                                # 这里只取hlist[0][x][x] 中的[0] 是为何？
                                delta_dis = hlist[0]['global_pos'][0] - tlist[0]['global_pos'][0]
                                if delta_dis < 0:
                                    ht_pair_distance[i, j] = -int(self.dis2idx[-delta_dis]) + self.dis_size // 2
                                else:
                                    ht_pair_distance[i, j] = int(self.dis2idx[delta_dis]) + self.dis_size // 2

                                j += 1 # 只有组成一对之后，j++
                    # 找出最大的一个 h_t_cnt， 这个 max_h_t_cnt 的最大值是930=31*31-31
                    max_h_t_cnt = max(max_h_t_cnt, j)
                    L_vertex.append(L)
                    titles.append(title)
                    entity_types.append(entity_type)
                    indexes.append(self.batches_order[idx][i])

            context_word_mask = context_word_ids > 0
            context_word_length = context_word_mask.sum(1)
            batch_max_length = context_word_length.max()

            yield {'context_idxs': get_cuda(context_word_ids[:cur_bsz, :batch_max_length].contiguous()),
                   'context_pos': get_cuda(context_pos_ids[:cur_bsz, :batch_max_length].contiguous()),
                   'context_ner': get_cuda(context_ner_ids[:cur_bsz, :batch_max_length].contiguous()),
                   'context_mention': get_cuda(context_mention_ids[:cur_bsz, :batch_max_length].contiguous()),
                   'context_word_mask': get_cuda(context_word_mask[:cur_bsz, :batch_max_length].contiguous()),
                   'context_word_length': get_cuda(context_word_length[:cur_bsz].contiguous()),
                   'h_t_pairs': get_cuda(ht_pairs[:cur_bsz, :max_h_t_cnt, :2]),
                   'relation_label': get_cuda(relation_label[:cur_bsz, :max_h_t_cnt]).contiguous(),
                   'relation_multi_label': get_cuda(relation_multi_label[:cur_bsz, :max_h_t_cnt]),
                   'relation_mask': get_cuda(relation_mask[:cur_bsz, :max_h_t_cnt]),
                   'ht_pair_distance': get_cuda(ht_pair_distance[:cur_bsz, :max_h_t_cnt]),
                   'labels': label_list,
                   'L_vertex': L_vertex,
                   'titles': titles,
                   'indexes': indexes,
                   'graphs': graph_list,
                   'entity2mention_table': entity2mention_table,
                   'entity_graphs': entity_graph_list,
                   'path_table': path_table,
                   'overlaps': overlaps,
                   'entity_types': entity_types
                   }
