import os
import pandas as pd
import torch


from argparse import ArgumentParser
from torch.utils.data import DataLoader

from utils import set_seed, parsing, DataPreprocess, Trainer
from utils import LogDatasetDomainAdaptation_Train, LogDatasetDomainAdaptation_Eval, LogDatasetDomainAdaptation_Test
from model import LogBERT_DA

def arg_parser():

    parser = ArgumentParser()

    parser.add_argument("--source_dataset_name", help="please choose source dataset name from BGL or Thunderbird", default="Thunderbird")
    parser.add_argument("--target_dataset_name", help="please choose target dataset name from BGL or Thunderbird", default="BGL")
    parser.add_argument("--device", help="hardware device", default='cuda')
    parser.add_argument("--random_seed", help="random seed", default=219)
    parser.add_argument("--download_datasets", help="download datasets or not", default=False)
    parser.add_argument("--output_dir", metavar="DIR", help="output directory", default="Dataset")
    parser.add_argument("--model_dir", metavar="DIR", help="output directory", default="Dataset")

    # data preprocess parameters
    parser.add_argument('--train_test_ratio', help='ratio of the size of train set to test set', default=0.8)
    parser.add_argument('--source_target_ratio', help='ratio of the size of source domain to target domain', default=100)
    parser.add_argument('--feature', help='feature fields for model input', default='EventTemplate')
    parser.add_argument('--minimum_drop_length', help='drop sequences whose length is less than the shreshold, valid only when minimum_drop_length > 0', default=5)
    
    # training parameters
    parser.add_argument("--max_epoch", help="epochs", default=2000)
    parser.add_argument("--batch_size", help="batch size", default=360)  # 1200, 600
    parser.add_argument("--lr", help="learning size", default=0.0001)
    parser.add_argument("--weight_decay", help="weight decay", default=1e-6)
    parser.add_argument("--eps", help="minimum center value", default=0.1)
    parser.add_argument("--n_epochs_stop", help="n epochs stop if not improve in valid loss", default=10)
    parser.add_argument("--loss_path", metavar="DIR", help="loss directory", default="loss_path")
    parser.add_argument("--model_path", metavar="DIR", help="saved model dir", default="model_path")
    parser.add_argument('--auto_mixed_precision', help='do amp or not', default=True)
    parser.add_argument('--if_step_lr', help='do weight decay or not', default=True)
    parser.add_argument('--lr_change_step', help='', default=1)
    parser.add_argument('--lr_change_gamma', help='', default=0.995)
    parser.add_argument('--patience', help='patience of early stop', default=500)

    # word2vec parameters
    parser.add_argument('--do_w2v', help='do word2vec embedding or not', default=False)
    parser.add_argument('--emb_dim', help='word2vec vector size', default=300)
    parser.add_argument('--min_count', help='minimum length of word2vec vocab', default=1)
    parser.add_argument('--w2c_workers', help='num of word2vec workers', default=8)
    
    # bert parameters
    parser.add_argument('--bert_pretrained_model', help='bert pretrained model', default=None)
    parser.add_argument('--bert_config_file_path', help='path of bert config file', default='./bert_tiny_checkpoint')
    parser.add_argument('--n_processer', help='num of bert tokenizer workers', default=0)
    parser.add_argument('--padding', help='do padding or not', default='max_length')
    parser.add_argument('--max_length', help='max length of sequence token', default=512)
    parser.add_argument('--class_pred_dim', help='class_pred_dim: dimension of class predict feedforward network', default=64)
    parser.add_argument('--domain_pred_dim', help='dimension of domain predict feedforward network', default=64)

    # data preprocessing parameters
    parser.add_argument("--window_size", help="size of sliding window", default=20)
    parser.add_argument("--step_size", help="step size of sliding window", default=4)

    # LSTM parameters
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

    return parser




def Fortnight():

    parser = arg_parser()
    args = parser.parse_args()
    options = vars(args)

    '''
    Random seed configuration -> default: 0219
    '''
    set_seed(options['random_seed'])
    print('\nset seed: 0{}'.format(options['random_seed']))

    '''
    File path configuration
    '''
    current_path = os.getcwd()

    loss_path = os.path.join(current_path, options['loss_path'])
    if not os.path.exists(loss_path):
        print('Making directory for loss storage: {}'.format(loss_path))
        os.mkdir(loss_path)

    model_path = os.path.join(current_path, options['model_path'])
    if not os.path.exists(model_path):
        print('Making directory for model checkpoint storage: {}\n'.format(model_path))
        os.mkdir(model_path)

    '''
    Dataset configuration
    '''
    dataset_root_path = './Dataset'

    if options['download_datasets'] == True:
        print('download and preprocess dataset\n')
        parsing(options['source_dataset_name'], output_dir=options['output_dir'])
        parsing(options['target_dataset_name'], output_dir=options['output_dir'])
    elif options['download_datasets'] == False:
        print('skip dataset download\n')

    # df_source = pd.read_csv(os.path.join(dataset_root_path, '{}.log_structured.csv'.format(options['source_dataset_name'])), nrows=10000)
    print('loading source dataset: {} dataset'.format(options['source_dataset_name']))

    # df_target = pd.read_csv(os.path.join(dataset_root_path, '{}.log_structured.csv'.format(options['target_dataset_name'])), nrows=1000)
    print('loading target dataset: {} dataset\n'.format(options['target_dataset_name']))

    source_df_path=os.path.join(dataset_root_path, '{}.log_structured.csv'.format(options['source_dataset_name']))
    target_df_path=os.path.join(dataset_root_path, '{}.log_structured.csv'.format(options['target_dataset_name']))

    '''
    FIXME: 新增 options 参数
    * do_w2v: 是否进行 word2vec 编码
    * minimum_drop_length: 长度丢弃阈值
    * feature: 模型的输入是 csv 文件中的哪一个字段, default='Content'
    '''
    dataset = DataPreprocess(source_df_path=source_df_path, target_df_path=target_df_path,
                             feature_field=options['feature'], minimum_drop_length=options['minimum_drop_length'],
                             do_w2v=options['do_w2v'], emb_dim=options['emb_dim'], min_count=options['min_count'],
                             seed=options['random_seed'], w2c_workers=options['w2c_workers'],
                             bert_pretrained_model=options['bert_pretrained_model'], bert_config_file_path=options['bert_config_file_path'],
                             n_processer=options['n_processer'], padding=options['padding'], max_length=options['max_length'],
                             window_size=options['window_size'], step_size=options['step_size'], 
                             train_size=options['train_test_ratio'], source_target_ratio=options['source_target_ratio'])
    train_dataset = LogDatasetDomainAdaptation_Train(source_data_dict=dataset.get_train_data_dict()[0],
                                                     target_data_dict=dataset.get_train_data_dict()[1])
    eval_dataset = LogDatasetDomainAdaptation_Eval(source_data_dict=dataset.get_eval_data_dict())
    test_dataset = LogDatasetDomainAdaptation_Test(target_data_dict=dataset.get_test_data_dict())

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=options['batch_size'], drop_last=False, pin_memory=True, num_workers=8)
    eval_loader = DataLoader(eval_dataset, shuffle=True, batch_size=options['batch_size'], drop_last=False, pin_memory=True, num_workers=8)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=options['batch_size'], drop_last=False, pin_memory=True, num_workers=8)

    model = LogBERT_DA(class_pred_dim=options['class_pred_dim'],
                       domain_pred_dim=options['domain_pred_dim'], 
                       bert_config_file_path=options['bert_config_file_path'])
    
    LogBERT_DA_trainer = Trainer(options=options, model=model)
    LogBERT_DA_trainer.train(options=options,
                             train_loader=train_loader,
                             eval_loader=eval_loader,
                             test_loader=test_loader)

    LogBERT_DA_trainer.test(weight_file_path='./model_path',  # options['model_path']
                            test_loader=test_loader)


    pass



if __name__ == '__main__':

    import warnings
    warnings.filterwarnings('ignore')
    
    torch.cuda.empty_cache() 

    Fortnight()