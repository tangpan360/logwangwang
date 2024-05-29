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


class Word2Vec_Embedding(object):

    def __init__(self, df_source: pd.DataFrame, df_target: pd.DataFrame,
                 feature_field: str = 'EventTemplate', emb_dim: int = 150,
                 min_count: int = 1, seed: int = 219, n_workers: int = 8) -> None:
        super(Word2Vec_Embedding, self).__init__()
        
        self.model = Word2Vec()
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.df_source = df_source
        self.df_target = df_target

        self.feature_field = feature_field
        self.emb_dim = emb_dim
        self.min_count = min_count
        self.seed = seed
        self.n_workers = n_workers

    def train(self, ):

        sentence_list = self.df_source[self.feature_field].to_list() + self.df_target[self.feature_field].to_list()
        token_sentence_list = list()

        for sentence in sentence_list:
            token_sentence_list.append([x.lower() for x in self.tokenizer.tokenize(sentence)])

        max_length = max(len(sentence) for sentence in token_sentence_list)
        print('maxmum sentence length: {}\n'.format(max_length))

        self.model = self.model(token_sentence_list, size=self.emb_dim, min_count=self.min_count, \
                                seed=self.seed, workers=self.n_workers)
    
    def get_sentence_emb(self, sentence: str):

        tokens_sentence = [x.lower() for x in self.tokenizer.tokenize(sentence)]
        emb_sentence_list = list()

        for token in tokens_sentence:
            corpus = list(self.model.wv.vocab.keys())

            if token in corpus:
                emb_sentence_list.append(self.model[token])
            else:
                self.model.build_vocab([token], update=True)
                self.model.train([token], epochs=1, total_examples=len([token]))

                emb_sentence_list.append(self.model[token])

        sen_emb = np.mean(np.array(emb_sentence_list), axis=0)

        return sen_emb
    
    def get_df_emb(self, get_source: bool = True):

        if get_source == True:
            self.df_source['Embedding'] = self.df_source[self.feature_field].apply(self.get_sentence_emb)

            return self.df_source
        else:
            self.df_target['Embedding'] = self.df_target[self.feature_field].apply(self.get_sentence_emb)

            return self.df_target

    def get_df_emb_DA_task(self, ):

        self.train()
        emb_source_df = self.get_df_emb(get_source=True)
        emb_target_df = self.get_df_emb(get_source=False)

        return emb_source_df, emb_target_df


class BERT_Tokenizer(object):

    def __init__(self, bert_pretrained_model: str = None, bert_config_file_path: str = None,
                 n_processer: int = 0, padding: str = 'max_length', max_length: int = 512) -> None:
        super(BERT_Tokenizer, self).__init__()

        if bert_pretrained_model != None:
            print('downloading bert tokenizer on Internet')
            self.tokenizer = BertTokenizer.from_pretrained(bert_pretrained_model)
        else:
            print('loading bert tokenizer locally\n')
            self.tokenizer = BertTokenizer.from_pretrained(bert_config_file_path)

        self.n_processer = n_processer
        self.padding = padding
        self.max_length = max_length

    def __bert_tokenizer_task(self, idx: int, split_sen_list: list):
        return (idx, self.tokenizer(split_sen_list, return_tensors='pt', 
                                    padding=self.padding, truncation=True, max_length=self.max_length, ))

    def get_sen_token(self, sentence_list: List[List[str]]):

        sentence_list = [' [SEP] '.join(sentence) for sentence in sentence_list]

        if self.n_processer > 1:
            per_part_size = int(len(sentence_list) / self.n_processer) - 1

            split_sen_list = list()
            for idx in range(0, self.n_processer * per_part_size, per_part_size):
                split_sen_list.append(sentence_list[idx: idx+per_part_size])
            
            # remain_sen_list = sentence_list[self.n_processer * per_part_size:]

            del sentence_list

            pool = ProcessPoolExecutor()
            token_list = [pool.submit(self.__bert_tokenizer_task, idx, item, self.tokenizer) for idx, item in enumerate(split_sen_list)]
            wait(token_list)

            result_list = list()
            for task in token_list:
                result_list.append(task.result()[1])

            # result_list.append(bert_tokenizer(remain_sen_list, return_tensors='pt', 
            #                                   padding=self.padding, truncation=True, max_length=self.max_length))
            final_result_dict = dict()
            final_result_dict['input_ids'] = torch.cat([res['input_ids'] for res in result_list], dim=0)
            final_result_dict['token_type_ids'] = torch.cat([res['token_type_ids'] for res in result_list], dim=0)
            final_result_dict['attention_mask'] = torch.cat([res['attention_mask'] for res in result_list], dim=0)

            return final_result_dict
        else:
            return self.tokenizer(sentence_list, return_tensors='pt', 
                                  padding=self.padding, truncation=True, max_length=self.max_length)


class DataPreprocess(object):

    def __init__(self, source_df_path: str, target_df_path: str, n_rows: int = 50000,
                 # word2vec parameters
                 feature_field: str = 'EventTemplate', minimum_drop_length: int = 5,
                 do_w2v: bool = True, emb_dim: int = 150, min_count: int = 1, seed: int = 219, w2c_workers: int = 8,
                 # bert parameters
                 bert_pretrained_model: str = None, bert_config_file_path: str = None,
                 n_processer: int = 0, padding: str = 'max_length', max_length: int = 512,
                 # sliding window parameters
                 window_size: int = 20, step_size: int = 4,
                 # train test parameters
                 train_size: float = 0.8, source_target_ratio: int = 100) -> None:
        super(DataPreprocess).__init__()

        # if n_rows > 0:
        #     df_source = pd.read_csv(source_df_path, nrows=n_rows)
        #     df_target = pd.read_csv(target_df_path, nrows=n_rows)
        # else:
        #     df_source = pd.read_csv(source_df_path, )
        #     df_target = pd.read_csv(target_df_path, )    

        '''
        Preprocess only
        '''
        # # TODO: 只写了 BERT 的数据预处理, word2vec 的还没写
        # s_log_dict, s_class_list, s_domain_list, \
        # t_log_dict, t_class_list, t_domain_list = self.get_datasets(df_source=df_source, df_target=df_target, 
        #                                                             feature_field=feature_field, minimum_drop_length=minimum_drop_length,
        #                                                             do_w2v=do_w2v, emb_dim=emb_dim, min_count=min_count, 
        #                                                             seed=seed, w2c_workers=w2c_workers,
        #                                                             bert_pretrained_model=bert_pretrained_model,
        #                                                             bert_config_file_path=bert_config_file_path,
        #                                                             n_processer=n_processer, padding=padding, max_length=max_length,
        #                                                             window_size=window_size, step_size=step_size)
        
        '''
        Jupyter only
        '''
        df_source = pd.read_feather('./Dataset/Thunderbird.log_structured_slided.feather', ).sample(frac=0.1, ignore_index=True, random_state=seed)
        df_target = pd.read_feather('./Dataset/BGL.log_structured_slided.feather', ).sample(frac=0.1, ignore_index=True, random_state=seed)

        # 从目标域抽取100组数据，扩充到和源域相等的数量
        df_label_0 = df_target[df_target['label'] == 0]
        df_label_1 = df_target[df_target['label'] == 1]
        assert len(df_label_0) >= 50, "label 为 0 的样本少于 50 个"
        assert len(df_label_1) >= 50, "label 为 1 的样本少于 50 个"
        df_label_0_sample = df_label_0.sample(n=50, random_state=42)
        df_label_1_sample = df_label_1.sample(n=50, random_state=42)
        df_sampled_target = pd.concat([df_label_0_sample, df_label_1_sample])
        df_target = df_target.drop(df_sampled_target.index)

        df_sampled_target = pd.concat([df_sampled_target] * 1000, ignore_index=True)
        # df_source = pd.concat((df_source, df_sampled_target), ignore_index=True)
        # df_source = df_source.sample(frac=1, random_state=seed, ignore_index=True)


        '''
        Jupyter only
        '''
        source_input_ids = torch.Tensor(np.vstack(df_source['input_ids'].values))
        source_token_type_ids = torch.zeros(source_input_ids.shape[0], 512)
        source_attention_mask = torch.Tensor(np.vstack(df_source['attention_mask'].values))
        s_class_list = list(df_source['label'].values)
        s_domain_list = [0] * source_input_ids.shape[0]

        '''
        Preprocess only
        '''
        # source_input_ids = s_log_dict['input_ids']
        # source_token_type_ids = s_log_dict['token_type_ids']
        # source_attention_mask = s_log_dict['attention_mask']

        s_ids_train, s_ids_eval, s_type_train, s_type_eval, s_mask_train, s_mask_eval, \
        s_class_train, s_class_eval, s_domain_train, s_domain_eval = train_test_split(source_input_ids, 
                                                                                      source_token_type_ids, 
                                                                                      source_attention_mask, 
                                                                                      s_class_list, 
                                                                                      s_domain_list, 
                                                                                      train_size=train_size, 
                                                                                      random_state=seed)
        self.source_input_ids_train = s_ids_train
        self.source_token_type_ids_train = s_type_train
        self.source_attention_mask_train = s_mask_train
        self.source_class_label_train = s_class_train
        self.source_domain_label_train = s_domain_train

        self.source_input_ids_eval = s_ids_eval
        self.source_token_type_ids_eval = s_type_eval
        self.source_attention_mask_eval = s_mask_eval
        self.source_class_label_eval = s_class_eval
        self.source_domain_label_eval = s_domain_eval

        '''
        FIXME: 新增 options 参数
        * source_target_ratio: 源域和目标域数据数量的比例
        '''
        '''
        Preprocess only
        '''
        # target_input_ids = t_log_dict['input_ids']
        # target_token_type_ids = t_log_dict['token_type_ids']
        # target_attention_mask = t_log_dict['attention_mask']

        '''
        Jupyter only
        '''
        target_input_ids = torch.Tensor(np.vstack(df_target['input_ids'].values))
        target_token_type_ids = torch.zeros(target_input_ids.shape[0], 512)
        target_attention_mask = torch.Tensor(np.vstack(df_target['attention_mask'].values))
        t_domain_list = [1] * target_input_ids.shape[0]

        # 1. 根据源域和目标域的比例确定采样比例
        source_target_ratio = source_target_ratio
        target_seq_sample_nums = int(s_ids_train.shape[0] / source_target_ratio)

        # 2. 从目标域数据中随机抽样 target_seq_nums 个样本
        '''
        TODO: 抽样时需要保证目标域数据集的长度大于采样长度 target_seq_sample_nums, 尝试用 try except 完善
        '''
        rondam_sample_index = torch.LongTensor(random.sample(range(target_input_ids.shape[0]), target_seq_sample_nums))
        
        target_input_ids_sample = torch.index_select(target_input_ids, 0, rondam_sample_index)
        target_token_type_sample = torch.index_select(target_token_type_ids, 0, rondam_sample_index)
        target_attention_mask_sample = torch.index_select(target_attention_mask, 0, rondam_sample_index)

        # 3. 将目标域抽样后得到的样本复制扩充 source_target_ratio + 1 倍
        target_input_ids_sample = target_input_ids_sample.repeat(source_target_ratio + 1, 1)
        target_token_type_sample = target_token_type_sample.repeat(source_target_ratio + 1, 1)
        target_attention_mask_sample = target_attention_mask_sample.repeat(source_target_ratio + 1, 1)

        # 4. 从扩充后的样本中随机抽样 s_ids_train.shape[0] (源域训练集数量) 个样本
        rondam_sample_index = torch.LongTensor(random.sample(range(target_input_ids_sample.shape[0]), s_ids_train.shape[0]))

        self.target_input_ids_train = torch.index_select(target_input_ids_sample, 0, rondam_sample_index)
        self.target_token_type_ids_train = torch.index_select(target_token_type_sample, 0, rondam_sample_index)
        self.target_attention_mask_train = torch.index_select(target_attention_mask_sample, 0, rondam_sample_index)
        self.target_domain_label_train = ((source_target_ratio + 1) * t_domain_list)[:s_ids_train.shape[0]]

        self.target_input_ids_train_with_label = torch.Tensor(np.vstack(df_sampled_target['input_ids'].values))
        self.target_token_type_ids_with_label = torch.zeros(self.target_input_ids_train.shape[0], 512)
        self.target_attention_mask_train_with_label = torch.Tensor(np.vstack(df_sampled_target['attention_mask'].values))
        self.target_class_label_train_with_label = list(df_sampled_target['label'].values)
        self.target_domain_label_train_with_label = [1] * self.target_input_ids_train.shape[0]

        '''
        将目标域上的全部数据作为测试集, 标签包含 class label 和 domain label
        '''
        '''
        Jupyter only
        '''
        self.target_input_ids_test = torch.Tensor(np.vstack(df_target['input_ids'].values))
        self.target_token_type_ids_test = torch.zeros(self.target_input_ids_test.shape[0], 512)
        self.target_attention_mask_test = torch.Tensor(np.vstack(df_target['attention_mask'].values))
        self.target_class_label_test = list(df_target['label'].values)
        self.target_domain_label_test = [1] * self.target_input_ids_test.shape[0]

        del df_source, df_target, s_ids_train, s_ids_eval, s_type_train, s_type_eval, s_mask_train, s_mask_eval, \
            s_class_train, s_class_eval, s_domain_train, s_domain_eval, target_input_ids, target_token_type_ids, \
            target_attention_mask, source_target_ratio, target_seq_sample_nums, rondam_sample_index, target_input_ids_sample, \
            target_token_type_sample, target_attention_mask_sample, df_sampled_target
        
    @staticmethod
    def minimum_length_drop(df: pd.DataFrame, feature: str, minimum_length: int = 5):
    
        df = df.loc[df[feature].str.len() > minimum_length].reset_index(drop=True)

        return df
    
    def sliding_window(self, df: pd.DataFrame, window_size:int = 20, step_size: int = 4,
                       target: int = 0, do_w2v: bool = True, feature_field: str = 'EventTemplate'):
    
        df['Label'] = df['Label'].apply(lambda x: int(x != '-'))

        log_seq_nums = df.shape[0]

        if do_w2v:
            log_key_list = list()
            log_key_emb_list = list()
            class_label_list = list()
            domain_label_list = list()

            for idx in np.arange(0, log_seq_nums, step=step_size, dtype=int):
                try:
                    log_key_list.append(df[feature_field].values[idx: idx+step_size].tolist())
                    log_key_emb_list.append(df['Embedding'].values[idx: idx+window_size])
                    class_label_list.append(max(df['Label'].values[idx: idx+window_size]))
                    domain_label_list.append(target)

                except IndexError:
                    pass
        else:
            log_key_list = list()
            class_label_list = list()
            domain_label_list = list()

            for idx in np.arange(0, log_seq_nums, step=step_size, dtype=int):
                try:
                    log_key_list.append(df[feature_field].values[idx: idx+step_size].tolist())
                    class_label_list.append(max(df['Label'].values[idx: idx+window_size]))
                    domain_label_list.append(target)

                except IndexError:
                    pass

        return (log_key_list, log_key_emb_list, class_label_list, domain_label_list) if do_w2v else \
               (log_key_list, class_label_list, domain_label_list)
    
    def get_datasets(self, df_source: pd.DataFrame, df_target: pd.DataFrame,
                     # word2vec parameters
                     feature_field: str = 'EventTemplate', minimum_drop_length: int = 5,
                     do_w2v: bool = True, emb_dim: int = 150, min_count: int = 1, seed: int = 219, w2c_workers: int = 8,
                     # bert parameters
                     bert_pretrained_model: str = None, bert_config_file_path: str = None,
                     n_processer: int = 0, padding: str = 'max_length', max_length: int = 512,
                     # sliding window parameters
                     window_size: int = 20, step_size: int = 4, ):
        
        print('window size: {}, step size: {}\n'.format(window_size, step_size))
    
        # TODO: 需要确认目标字段在原始 csv 文件中是否有缺失值
        df_source[feature_field].fillna('EmptyParametersTokens', inplace=True)
        df_target[feature_field].fillna('EmptyParametersTokens', inplace=True)

        if minimum_drop_length > 0:
            df_source = self.minimum_length_drop(df=df_source, feature=feature_field, minimum_length=minimum_drop_length)
            df_target = self.minimum_length_drop(df=df_target, feature=feature_field, minimum_length=minimum_drop_length)

        # do word2vec embedding or bert tokenizer
        if do_w2v:
            start_time = time.time()
            word2vec_emb = Word2Vec_Embedding(df_source=df_source, df_target=df_target,
                                              feature_field=feature_field, emb_dim=emb_dim, min_count=min_count, 
                                              seed=seed, n_workers=w2c_workers)
            df_source, df_target = word2vec_emb.get_df_emb_DA_task()
            print('word2vec train time use: {} s'.format(time.time() - start_time))
            
            df_source = df_source[['Label', feature_field, 'Embedding', ]]
            df_target = df_target[['Label', feature_field, 'Embedding', ]]

            start_time = time.time()
            source_log_key_list, source_key_emb_list, source_class_label_list, source_domain_label_list \
                = self.sliding_window(df=df_source, window_size=window_size, step_size=step_size, target=0,
                                      do_w2v=do_w2v, feature_field=feature_field)
            target_log_key_list, target_key_emb_list, target_class_label_list, target_domain_label_list \
                = self.sliding_window(df=df_target, window_size=window_size, step_size=step_size, target=1,
                                      do_w2v=do_w2v, feature_field=feature_field)
            print('sliding windows time use: {} s\n'.format(time.time() - start_time))

            return source_key_emb_list, source_class_label_list, source_domain_label_list, \
                   target_key_emb_list, target_class_label_list, target_domain_label_list
        else:
            bert_tokenizer = BERT_Tokenizer(bert_pretrained_model=bert_pretrained_model, bert_config_file_path=bert_config_file_path,
                                            n_processer=n_processer, padding=padding, max_length=max_length)
            
            df_source = df_source[['Label', feature_field, ]]
            df_target = df_target[['Label', feature_field, ]]

            start_time = time.time()
            source_log_key_list, source_class_label_list, source_domain_label_list = self.sliding_window(df=df_source, 
                                                                                                         window_size=window_size, 
                                                                                                         step_size=step_size, 
                                                                                                         target=0,
                                                                                                         do_w2v=do_w2v, 
                                                                                                         feature_field=feature_field)
            target_log_key_list, target_class_label_list, target_domain_label_list = self.sliding_window(df=df_target, 
                                                                                                         window_size=window_size, 
                                                                                                         step_size=step_size, 
                                                                                                         target=1,
                                                                                                         do_w2v=do_w2v, 
                                                                                                         feature_field=feature_field)
            print('sliding windows time use: {} s'.format(time.time() - start_time))
            start_time = time.time()
            source_log_key_token_dict = bert_tokenizer.get_sen_token(source_log_key_list)
            print('source data tokenizer time use: {} s'.format(time.time() - start_time))
        
            start_time = time.time()
            target_log_key_token_dict = bert_tokenizer.get_sen_token(target_log_key_list)
            print('target data tokenizer time use: {} s\n'.format(time.time() - start_time))

            return source_log_key_token_dict, source_class_label_list, source_domain_label_list, \
                   target_log_key_token_dict, target_class_label_list, target_domain_label_list
               

    def get_train_data_dict(self, ):
        return {'input_ids': self.source_input_ids_train, 'attention_mask': self.source_attention_mask_train, \
                'class_label': self.source_class_label_train, 'domain_label': self.source_domain_label_train}, \
               {'input_ids': self.target_input_ids_train, 'attention_mask': self.target_attention_mask_train, \
                'domain_label': self.target_domain_label_train}, \
               {'input_ids': self.target_input_ids_train_with_label,
                'attention_mask': self.target_attention_mask_train_with_label,
                'class_label': self.target_class_label_train_with_label,
                'domain_label': self.target_domain_label_train_with_label}

    def get_eval_data_dict(self, ):
        return {'input_ids': self.source_input_ids_eval, 'attention_mask': self.source_attention_mask_eval, \
                'class_label': self.source_class_label_eval, 'domain_label': self.source_domain_label_eval}
    
    def get_test_data_dict(self, ):
        return {'input_ids': self.target_input_ids_test, 'attention_mask': self.target_attention_mask_test, \
                'class_label': self.target_class_label_test, 'domain_label': self.target_domain_label_test}
            

class LogDatasetDomainAdaptation_Train(Dataset):

    def __init__(self, source_data_dict: dict, target_data_dict: dict, target_data_with_label_dict: dict) -> None:
        super(LogDatasetDomainAdaptation_Train).__init__()

        self.source_input_ids_train: torch.Tensor = source_data_dict['input_ids']
        self.source_attention_mask_train: torch.Tensor = source_data_dict['attention_mask']
        self.source_class_label_train: list = source_data_dict['class_label']
        self.source_domain_label_train: list = source_data_dict['domain_label']

        self.target_input_ids_train: torch.Tensor = target_data_dict['input_ids']
        self.target_attention_mask_train: torch.Tensor = target_data_dict['attention_mask']
        self.target_domain_label_train: list = target_data_dict['domain_label']

        self.target_input_ids_train_with_label: torch.Tensor = target_data_with_label_dict['input_ids']
        self.target_attention_mask_train_with_label: torch.Tensor = target_data_with_label_dict['attention_mask']
        self.target_class_label_train_with_label: list = target_data_with_label_dict['class_label']
        self.target_domain_label_train_with_label: list = target_data_with_label_dict['domain_label']

    def __getitem__(self, idx) -> Tuple[dict]:
        
        source_input_ids = self.source_input_ids_train[idx].long()
        source_attention_mask = self.source_attention_mask_train[idx].long()
        source_class_label = torch.Tensor([self.source_class_label_train[idx]]).float()
        source_domain_label = torch.Tensor([self.source_domain_label_train[idx]]).float()

        target_input_ids =  self.target_input_ids_train[idx].long()
        target_attention_mask = self.target_attention_mask_train[idx].long()
        target_domain_label = torch.Tensor([self.target_domain_label_train[idx]]).float()

        target_input_ids_with_label = self.target_input_ids_train_with_label[idx].long()
        target_attention_mask_with_label = self.target_attention_mask_train_with_label[idx].long()
        target_class_label_with_label = torch.Tensor([self.target_class_label_train_with_label[idx]]).float()
        target_domain_label_with_label = torch.Tensor([self.target_domain_label_train_with_label[idx]]).float()

        return {'input_ids': source_input_ids, 'attention_mask': source_attention_mask}, \
               {'class_label': source_class_label, 'domain_label': source_domain_label}, \
               {'input_ids': target_input_ids, 'attention_mask': target_attention_mask}, \
               {'domain_label': target_domain_label}, \
               {'input_ids': target_input_ids_with_label, 'attention_mask': target_attention_mask_with_label}, \
               {'class_label': target_class_label_with_label, 'domain_label': target_domain_label_with_label}
    
    def __len__(self, ):
        return self.source_input_ids_train.shape[0]
    

class LogDatasetDomainAdaptation_Eval(Dataset):

    def __init__(self, source_data_dict: dict)  -> None:
        super(LogDatasetDomainAdaptation_Eval).__init__()

        self.source_input_ids_eval: torch.Tensor = source_data_dict['input_ids']
        self.source_attention_mask_eval: torch.Tensor = source_data_dict['attention_mask']
        self.source_class_label_eval: list = source_data_dict['class_label']
        self.source_domain_label_eval: list = source_data_dict['domain_label']

    def __getitem__(self, idx) -> Tuple[dict]:

        source_input_ids = self.source_input_ids_eval[idx].long()
        source_attention_mask = self.source_attention_mask_eval[idx].long()
        source_class_label = torch.Tensor([self.source_class_label_eval[idx]]).float()
        source_domain_label = torch.Tensor([self.source_domain_label_eval[idx]]).float()

        return {'input_ids': source_input_ids, 'attention_mask': source_attention_mask}, \
               {'class_label': source_class_label, 'domain_label': source_domain_label}
    
    def __len__(self, ):
        return self.source_input_ids_eval.shape[0]
    

class LogDatasetDomainAdaptation_Test(Dataset):

    def __init__(self, target_data_dict: dict) -> None:
        super(LogDatasetDomainAdaptation_Test).__init__()

        self.target_input_ids_test: torch.Tensor = target_data_dict['input_ids']
        self.target_attention_mask_test: torch.Tensor = target_data_dict['attention_mask']
        self.target_class_label_test: list = target_data_dict['class_label']
        self.target_domain_label_test: list = target_data_dict['domain_label']

    def __getitem__(self, idx) -> Tuple[dict]:

        target_input_ids =  self.target_input_ids_test[idx].long()
        target_attention_mask = self.target_attention_mask_test[idx].long()
        target_class_label = torch.Tensor([self.target_class_label_test[idx]]).float()
        target_domain_label = torch.Tensor([self.target_domain_label_test[idx]]).float()

        return {'input_ids': target_input_ids, 'attention_mask': target_attention_mask}, \
               {'class_label': target_class_label, 'domain_label': target_domain_label}
    
    def __len__(self, ):
        return self.target_input_ids_test.shape[0]


if __name__ == '__main__':
    from argparse import ArgumentParser

    def arg_parser():
        """
        Add parser parameters
        :return:
        """
        parser = ArgumentParser()
        parser.add_argument("--source_dataset_name", help="please choose source dataset name from BGL or Thunderbird", default="Thunderbird")
        # parser.add_argument("--source_dataset_name", help="please choose source dataset name from BGL or Thunderbird", default="BGL")
        parser.add_argument("--target_dataset_name", help="please choose target dataset name from BGL or Thunderbird", default="BGL")
        # parser.add_argument("--target_dataset_name", help="please choose target dataset name from BGL or Thunderbird", default="Thunderbird")
        parser.add_argument("--device", help="hardware device", default="cpu")
        parser.add_argument("--random_seed", help="random seed", default=219)
        parser.add_argument("--download_datasets", help="download datasets or not", default=False)
        parser.add_argument("--output_dir", metavar="DIR", help="output directory", default="Dataset")
        parser.add_argument("--model_dir", metavar="DIR", help="output directory", default="Dataset")

        # training parameters
        parser.add_argument("--max_epoch", help="epochs", default=100)
        parser.add_argument("--batch_size", help="batch size", default=2)
        parser.add_argument("--lr", help="learning size", default=0.001)
        parser.add_argument("--weight_decay", help="weight decay", default=1e-6)
        parser.add_argument("--eps", help="minimum center value", default=0.1)
        parser.add_argument("--n_epochs_stop", help="n epochs stop if not improve in valid loss", default=10)
        parser.add_argument("--loss_path", metavar="DIR", help="loss directory", default="loss_path")
        parser.add_argument("--model_path", metavar="DIR", help="saved model dir", default="model_path")

        # word2vec parameters
        parser.add_argument("--emb_dim", help="word2vec vector size", default=300)

        # data preprocessing parameters
        parser.add_argument("--window_size", help="size of sliding window", default=20)
        parser.add_argument("--step_size", help="step size of sliding window", default=4)
        parser.add_argument("--train_size_s", help="source training size", default=100000)
        parser.add_argument("--train_size_t", help="target training size", default=1000)

        # LSTM parameters
        # parser.add_argument("--hid_dim", help="hidden dimensions", default=128)
        parser.add_argument("--hid_dim", help="hidden dimensions", default=768)
        parser.add_argument("--out_dim", help="output dimensions", default=2)
        parser.add_argument("--n_layers", help="layers of LSTM", default=2)
        parser.add_argument("--dropout", help="dropout", default=0.3)
        parser.add_argument("--bias", help="bias for LSTM", default=True)

        # gradient reversal parameters
        parser.add_argument("--alpha", help="alpha value for the gradient reversal layer", default=0.1)

        # test parameters
        parser.add_argument("--test_ratio", help="testing ratio", default=0.1)
        '''
        新增 options
        '''
        parser.add_argument('--do_w2v', help='do word2vec embedding or not', default=False)
        parser.add_argument('--minimum_drop_length', help='drop sequences whose length is less than the shreshold, valid only when minimum_drop_length > 0', default=5)
        parser.add_argument('--feature', help='feature fields for model input', default='Content')
        parser.add_argument('--train_test_ratio', help='ratio of the size of train set to test set', default=0.8)
        parser.add_argument('--source_target_ratio', help='ratio of the size of source domain to target domain', default=100)

        return parser
    
    parser = arg_parser()
    args = parser.parse_args()
    options = vars(args)

    '''
    './Dataset/Thunderbird.log_structured.csv'
    './Dataset/BGL.log_structured.csv'
    '''

    dataset = DataPreprocess(source_df_path='./Dataset/Thunderbird.log_structured.csv',
                             target_df_path='./Dataset/BGL.log_structured.csv',
                             options=options)
    
    train_dataset = LogDatasetDomainAdaptation_Train(source_data_dict=dataset.get_train_data_dict()[0],
                                                     target_data_dict=dataset.get_train_data_dict()[1])
    eval_dataset  = LogDatasetDomainAdaptation_Eval (source_data_dict=dataset.get_eval_data_dict())
    test_dataset  = LogDatasetDomainAdaptation_Test (target_data_dict=dataset.get_test_data_dict())
    

    pass
