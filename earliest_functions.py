import pandas as pd
import numpy as np
import time
import torch
import random

from gensim.models import Word2Vec
from nltk.tokenize import RegexpTokenizer
from transformers import BertTokenizer
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor, wait


'''
Word2Vector embedding functions
'''
def word2vec_train(df_source: pd.DataFrame, df_target: pd.DataFrame = None, 
                   feature: str = 'Content', emb_dim: int = 150, 
                   min_count: int = 1, seed: int = 219, workers: int = 8):
    
    sentence_list = df_source[feature].to_list() + df_target[feature].to_list()
    tokenizer = RegexpTokenizer(r'\w+')
    token_sentence_list = list()

    for sentence in sentence_list:
        token_sentence_list.append([x.lower() for x in tokenizer.tokenize(sentence)])

    max_length = max(len(sentence) for sentence in token_sentence_list)
    print('Max sentence length: {}\n'.format(max_length))

    word2vec = Word2Vec(token_sentence_list, size=emb_dim, min_count=min_count, seed=seed, workers=workers)

    return word2vec


def word2vec_get_sent_emb(df: pd.DataFrame, w2v: Word2Vec, feature: str = 'Content'):

    tokenizer = RegexpTokenizer(r'\w+')

    # TODO: 多线程访问的过程需要考虑多线程访问安全问题
    def sentence_emb(sentence: str): 
        tokens_sentence = [x.lower() for x in tokenizer.tokenize(sentence)]
        emb_list = list()

        for idx, token in enumerate(tokens_sentence):
            corpus = list(w2v.wv.vocab.keys())

            if token in corpus:
                emb_list.append(w2v[token])
            else:
                w2v.build_vocab([token], update=True)
                w2v.train([token], epochs=1, total_examples=len([token]))

                emb_list.append(w2v[token])

        sen_emb = np.mean(np.array(emb_list), axis=0)

        return sen_emb
    
    df['Embedding'] = df[feature].apply(sentence_emb)

    return df, w2v


def word2vec_emb(df_source: pd.DataFrame, df_target: pd.DataFrame,
                 emb_dim: int = 150, seed: int = 219, feature: str = 'Content'):

    w2v = word2vec_train(df_source=df_source, df_target=df_target, 
                         feature=feature, emb_dim=emb_dim, seed=seed, )
    
    df_source, w2v = word2vec_get_sent_emb(df=df_source, w2v=w2v)
    df_target, w2v = word2vec_get_sent_emb(df=df_target, w2v=w2v)

    return df_source, df_target, w2v

'''
BERT tokenizer functions
'''
def bert_tokenizer_task(idx: int, split_sen_list: list, bert_tokenizer: BertTokenizer):
    return (idx, bert_tokenizer(split_sen_list, return_tensors='pt', padding='max_length', truncation=True, max_length=512, ))

def bert_get_sen_token(sentence_list: List[List[str]], bert_tokenizer: BertTokenizer):

    n_processer = 10
    per_part_size = int(len(sentence_list) / n_processer)

    split_sen_list = list()
    for idx in range(0, n_processer * per_part_size, per_part_size):
        split_sen_list.append(sentence_list[idx: idx+per_part_size])
    
    remain_sen_list = sentence_list[n_processer * per_part_size:]

    del sentence_list

    pool = ProcessPoolExecutor()
    token_list = [pool.submit(bert_tokenizer_task, idx, item, bert_tokenizer) for idx, item in enumerate(split_sen_list)]
    wait(token_list)

    result_list = list()
    for task in token_list:
        result_list.append(task.result()[1])

    # result_list.append(bert_tokenizer(remain_sen_list, return_tensors='pt', padding=True, truncation=True, max_length=512))
    final_result_dict = dict()
    final_result_dict['input_ids'] = torch.cat([res['input_ids'] for res in result_list], dim=0)
    final_result_dict['token_type_ids'] = torch.cat([res['token_type_ids'] for res in result_list], dim=0)
    final_result_dict['attention_mask'] = torch.cat([res['attention_mask'] for res in result_list], dim=0)

    return final_result_dict

    # return bert_tokenizer(sentence_list, return_tensors='pt', padding=True, truncation=True, max_length=512)

# region
# # FIXME: 数据处理过程不能放在类里面, 实例化对象时会多次调用 bert_tokenizer 方法
# class LogDatasetDomainAdaptation(Dataset):

#     def __init__(self, source_df_path: str, target_df_path: str, options: dict, ) -> None:
#         super(LogDatasetDomainAdaptation).__init__()

#         df_source = pd.read_csv(source_df_path, nrows=1000)
#         df_target = pd.read_csv(target_df_path, nrows=10)

#         # source_train_size = options['train_size_s']
#         # target_train_size = options['train_size_t']
#         # source_target_ratio = int(source_train_size / target_train_size)

#         s_log_dict, s_class_list, s_domain_list, t_log_dict, t_class_list, t_domain_list = get_datasets(df_source=df_source, 
#                                                                                                         df_target=df_target, 
#                                                                                                         options=options)

#         source_input_ids = s_log_dict['input_ids']
#         source_token_type_ids = s_log_dict['token_type_ids']
#         aource_attention_mask = s_log_dict['attention_mask']

#         s_ids_train, s_ids_eval, s_type_train, s_type_eval, s_mask_train, s_mask_eval, \
#             s_class_train, s_class_eval, s_domain_train, s_domain_eval = train_test_split(source_input_ids, 
#                                                                                           source_token_type_ids, 
#                                                                                           aource_attention_mask, 
#                                                                                           s_class_list, 
#                                                                                           s_domain_list, 
#                                                                                           train_size=options['train_test_ratio'], 
#                                                                                           random_state=options['random_seed'])
        
#         self.source_input_ids_train = s_ids_train
#         self.source_token_type_ids_train = s_type_train
#         self.source_attention_mask_train = s_mask_train
#         self.source_class_label_train = s_class_train
#         self.source_domain_label_train = s_domain_train

#         self.source_input_ids_eval = s_ids_eval
#         self.source_token_type_ids_eval = s_type_eval
#         self.source_attention_mask_eval = s_mask_eval
#         self.source_class_label_eval = s_class_eval
#         self.source_domain_label_eval = s_domain_eval

#         '''
#         FIXME: 新增 options 参数
#         * source_target_ratio: 源域和目标域数据数量的比例
#         '''
#         target_input_ids = t_log_dict['input_ids']
#         target_token_type_ids = t_log_dict['token_type_ids']
#         target_attention_mask = t_log_dict['attention_mask']

#         # 1. 根据源域和目标域的比例确定采样比例
#         source_target_ratio = options['source_target_ratio']
#         target_seq_nums = int(s_ids_train.shape[0] / source_target_ratio)

#         # 2. 从目标域数据中随机抽样 target_seq_nums 个样本
#         rondam_sample_index = torch.LongTensor(random.sample(range(target_input_ids.shape[0]), target_seq_nums))
        
#         target_input_ids_sample = torch.index_select(target_input_ids, 0, rondam_sample_index)
#         target_token_type_sample = torch.index_select(target_token_type_ids, 0, rondam_sample_index)
#         target_attention_mask_sample = torch.index_select(target_attention_mask, 0, rondam_sample_index)

#         # 3. 将目标域抽样后得到的样本复制扩充 source_target_ratio + 1 倍
#         target_input_ids_sample = target_input_ids_sample.repeat(source_target_ratio + 1, 1)
#         target_token_type_sample = target_token_type_sample.repeat(source_target_ratio + 1, 1)
#         target_attention_mask_sample = target_attention_mask_sample.repeat(source_target_ratio + 1, 1)

#         # 4. 从扩充后的样本中随机抽样 s_ids_train.shape[0] (源域训练集数量) 个样本
#         rondam_sample_index = torch.LongTensor(random.sample(range(target_input_ids_sample.shape[0]), s_ids_train.shape[0]))

#         self.target_input_ids_train = torch.index_select(target_input_ids_sample, 0, rondam_sample_index)
#         self.target_token_type_ids_train = torch.index_select(target_token_type_sample, 0, rondam_sample_index)
#         self.target_attention_mask_train = torch.index_select(target_attention_mask_sample, 0, rondam_sample_index)
#         self.target_domain_label_train = ((source_target_ratio + 1) * t_domain_list)[:s_ids_train.shape[0]]

#         '''
#         将目标域上的全部数据作为测试集, 标签包含 class label 和 domain label
#         '''
#         self.target_input_ids_test = t_log_dict['input_ids']
#         self.target_token_type_ids_test = t_log_dict['token_type_ids']
#         self.target_attention_mask_test = t_log_dict['attention_mask']
#         self.target_class_label_test = t_class_list
#         self.target_domain_label_test = t_domain_list

#         del df_source, df_target, s_ids_train, s_ids_eval, s_type_train, s_type_eval, s_mask_train, s_mask_eval, \
#             s_class_train, s_class_eval, s_domain_train, s_domain_eval, target_input_ids, target_token_type_ids, \
#             target_attention_mask, source_target_ratio, target_seq_nums, rondam_sample_index, target_input_ids_sample, \
#             target_token_type_sample, target_attention_mask_sample

#     # FIXME: 以字典的形式传递训练时所需要的序列，但是仍然不够优雅，考虑重写 BatchSampler
#     def __getitem__(self, idx):

#         source_input_ids_train = self.source_input_ids_train[idx].long()
#         source_token_type_ids_train = self.source_token_type_ids_train[idx].long()
#         source_attention_mask_train = self.source_attention_mask_train[idx].long()
#         source_class_label_train = torch.Tensor([self.source_class_label_train[idx]]).long()
#         source_domain_label_train = torch.Tensor([self.source_domain_label_train[idx]]).long()

#         target_input_ids_train =  self.target_input_ids_train[idx].long()
#         target_token_type_ids_train = self.target_token_type_ids_train[idx].long()
#         target_attention_mask_train = self.target_attention_mask_train[idx].long()
#         target_domain_label_train = torch.Tensor([self.target_domain_label_train[idx]]).long()

#         if self.do_train and self.source:
#             source_input_ids = self.source_input_ids_train[idx].long()
#             source_token_type_ids = self.source_token_type_ids_train[idx].long()
#             source_attention_mask = self.source_attention_mask_train[idx].long()
#             source_class_label = torch.Tensor([self.source_class_label_train[idx]]).long()
#             source_domain_label = torch.Tensor([self.source_domain_label_train[idx]]).long()

#             target_input_ids =  self.target_input_ids_train[idx].long()
#             target_token_type_ids = self.target_token_type_ids_train[idx].long()
#             target_attention_mask = self.target_attention_mask_train[idx].long()
#             target_domain_label = torch.Tensor([self.target_domain_label_train[idx]]).long()

#             return (source_input_ids, source_token_type_ids, source_attention_mask), source_class_label, source_domain_label, \
#                    (target_input_ids, target_token_type_ids, target_attention_mask), target_domain_label
        
#         elif not self.do_train and self.source:
#             source_input_ids = self.source_input_ids_eval[idx].long()
#             source_token_type_ids = self.source_token_type_ids_eval[idx].long()
#             source_attention_mask = self.source_attention_mask_eval[idx].long()
#             source_class_label = torch.Tensor([self.source_class_label_eval[idx]]).long()
#             source_domain_label = torch.Tensor([self.source_domain_label_eval[idx]]).long()

#             return (source_input_ids, source_token_type_ids, source_attention_mask), source_class_label, source_domain_label

#         elif not self.do_train and not self.source:
#             target_input_ids =  self.target_input_ids_test[idx].long()
#             target_token_type_ids = self.target_token_type_ids_test[idx].long()
#             target_attention_mask = self.target_attention_mask_test[idx].long()
#             target_class_label = torch.Tensor([self.target_domain_label_test[idx]]).long()
#             target_domain_label = torch.Tensor([self.target_domain_label_test[idx]]).long()

#             return (target_input_ids, target_token_type_ids, target_attention_mask), target_class_label, target_domain_label

#     def __len__(self, ):

#         if self.do_train and self.source:
#             return self.source_input_ids_train.shape[0]
        
#         elif not self.do_train and self.source: 
#             return self.source_input_ids_eval.shape[0]
        
#         elif not self.do_train and not self.source:
#             return self.target_input_ids_test.shape[0]
# endregion