import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib
import os


from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.cuda import amp
from tqdm import tqdm
from utils import EarlyStopping, BCEFocalLoss
from utils import LogDatasetDomainAdaptation_Train, LogDatasetDomainAdaptation_Eval, LogDatasetDomainAdaptation_Test


class Trainer(object):
    
    def __init__(self, options: dict,
                 model: nn.Module) -> None:
        super(Trainer, self, ).__init__()

        self.device = options['device']

        '''
        Init model parameters
        '''  
        self.model = model.to(self.device)

        for param in self.model.parameters():
            param.requires_grad = True

        '''
        Init optimizer parameters and loss calculator

        FIXME: 新增 options 参数
        * auto_mixed_precision: 是否开启混合精度训练, default = True
        * if_step_lr: 是否进行学习率自动阶段更新, default = True
        * lr_change_step: 学习率每经过多少个 epoch 后改变
        * lr_change_gamma: 学习率改变的比例

        TODO: 
        * loss 暂定为 CrossEntropyLoss, 后期优化时尝试更换 loss function
        '''
        self.optimizer = optim.Adam(self.model.parameters(), lr=options['lr'])  # , weight_decay=options['weight_decay'])
        self.loss_calculator = BCEFocalLoss().to(self.device)
        self.amp = options['auto_mixed_precision']
        self.scaler = amp.GradScaler(enabled=self.amp)
        self.if_step_lr = options['if_step_lr']
        self.scheduler = optim.lr_scheduler.StepLR(optimizer=self.optimizer,
                                                   step_size=options['lr_change_step'],
                                                   gamma=options['lr_change_gamma'])

        # print('{}\n'.format(self.scheduler.get_last_lr()[0]))

        '''
        File path

        FIXME: options 新增参数
        * loss_save_path: 训练, 验证, 测试损失保存地址
        * model_save_path: 模型权重文件保存地址
        '''
        self.loss_save_path = options['loss_path']
        self.model_save_path = options['model_path']

        '''
        Others

        FIXME: options 新增参数
        * patience: EarlyStop 的忍耐次数, 当超过相应的 epoch 后模型在验证集上的预测效果没有提升, 则停止训练过陈
        '''
        self.early_stop = EarlyStopping(patience=options['patience'])

        # FIXME: 只保存 list, 其余误差值通过打印显示出来
        self.train_class_loss_list = list()
        self.train_doamin_loss_list = list()
        self.eval_class_loss_list = list()
        self.eval_doamin_loss_list = list()
        self.eval_loss_list = list()

        self.class_accuracy_list = list()
        self.class_precision_list = list()
        self.class_recall_list = list()
        self.class_f1_score_list = list()

        self.domain_accuracy_list = list()
        self.domain_precision_list = list()
        self.domain_recall_list = list()
        self.domain_f1_score_list = list()


    def _train_one_epoch(self, train_loader: LogDatasetDomainAdaptation_Train, 
                         eval_loader: LogDatasetDomainAdaptation_Eval, 
                         test_loader: LogDatasetDomainAdaptation_Test,
                         epoch: int,
                         epochs: int) -> None:

        self.train_class_loss = 0.
        self.train_doamin_loss = 0.
        self.eval_class_loss = 0.
        self.eval_doamin_loss = 0.
        eval_loss = 0.

        '''
        Train model
        '''
        self.model.train()
        loss = None

        for (i, batch) in enumerate(train_loader):

            p = float(i + (epoch - 1) * len(train_loader)) / epochs / len(train_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            source_data_train = batch[0]
            source_label_train = batch[1]
            # target_data_train = batch[2]
            # target_label_train = batch[3]
            # target_data_train_with_label = batch[4]
            # target_label_train_with_label = batch[5]
            
            self.optimizer.zero_grad()

            '''
            FIXME: 当前的模型推理过程较为特化, 其他模型不适用于此框架, 后期可围绕高扩展性重构此部分
            '''
            # 三种不同的损失 - source domain: 1. class loss, 2. domain loss; target domain: 3. domain loss 
            with amp.autocast(enabled=self.amp):

                '''
                w domain
                '''
                class_output, domain_output = self.model(input_ids=source_data_train['input_ids'].to(self.device),
                                                         attention_mask=source_data_train['attention_mask'].to(self.device),
                                                         token_type_ids=None,
                                                         alpha=alpha)
                source_class_loss = self.loss_calculator(class_output, source_label_train['class_label'].to(self.device))
                source_domain_loss = self.loss_calculator(domain_output, source_label_train['domain_label'].to(self.device))

                # class_output_target, _ = self.model(input_ids=target_data_train_with_label['input_ids'].to(self.device),
                #                                     attention_mask=target_data_train_with_label['attention_mask'].to(self.device))
                # class_output_target, domain_output_target = self.model(input_ids=target_data_train_with_label['input_ids'].to(self.device),
                #                                     attention_mask=target_data_train_with_label['attention_mask'].to(self.device))
                # target_class_loss_with_label = self.loss_calculator(class_output_target, target_label_train_with_label['class_label'].to(self.device))
                # target_domain_loss_with_label = self.loss_calculator(domain_output_target, target_label_train_with_label['domain_label'].to(self.device))

                # _, domain_output = self.model(input_ids=target_data_train['input_ids'].to(self.device),
                #                               attention_mask=target_data_train['attention_mask'].to(self.device))
                # target_domain_loss = self.loss_calculator(domain_output, target_label_train['domain_label'].to(self.device))

                # a = 0.9
                # loss =  a * source_class_loss + (1 - a) * (source_domain_loss + target_domain_loss)

                # loss = source_class_loss / (source_class_loss.detach() + 10e-8) + \
                #        source_domain_loss / (source_domain_loss.detach() + 10e-8) + \
                #        target_domain_loss / (target_domain_loss.detach() + 10e-8)
                # loss = source_class_loss + source_domain_loss + target_domain_loss + target_class_loss_with_label  # add by tp
                loss = source_class_loss + source_domain_loss  # add by tp
                # self.train_class_loss += source_class_loss.item() / len(train_loader)
                self.train_class_loss += source_class_loss.item() / len(train_loader)
                # self.train_doamin_loss += (source_domain_loss.item() + target_domain_loss.item()) / len(train_loader)
                self.train_doamin_loss += source_domain_loss.item() / len(train_loader)

                '''
                wo domain
                '''
                # class_output = self.model(input_ids=source_data_train['input_ids'].to(self.device),
                #                           attention_mask=source_data_train['attention_mask'].to(self.device))
                # source_class_loss = self.loss_calculator(class_output, source_label_train['class_label'].to(self.device))
                #
                # loss = source_class_loss
                # self.train_class_loss += source_class_loss.item() / len(train_loader)
  
            self.scaler.scale(loss).backward()
            # TODO 调整正则
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.8, norm_type=2)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        '''
        wo domain
        '''
        # self.train_class_loss_list.append(self.train_class_loss)

        '''
        w domain
        '''
        self.train_class_loss_list.append(self.train_class_loss)
        self.train_doamin_loss_list.append(self.train_doamin_loss)

        if self.if_step_lr:
            self.scheduler.step()

        '''
        Evaluate model       
        '''
        self.model.eval()

        with torch.no_grad():
            for batch in eval_loader:
                source_data_eval = batch[0]
                source_label_eval = batch[1]

                '''
                w domain 
                '''
                class_output, domain_output = self.model(input_ids=source_data_eval['input_ids'].to(self.device),
                                                         attention_mask=source_data_eval['attention_mask'].to(self.device),
                                                         token_type_ids=None,
                                                         alpha=0)
                source_class_loss = self.loss_calculator(class_output, source_label_eval['class_label'].to(self.device))
                source_domain_loss = self.loss_calculator(domain_output, source_label_eval['domain_label'].to(self.device))

                self.eval_class_loss += source_class_loss.item() / len(eval_loader)
                self.eval_doamin_loss += source_domain_loss.item() / len(eval_loader)

                '''
                wo domain
                '''
                # class_output = self.model(input_ids=source_data_eval['input_ids'].to(self.device),
                #                           attention_mask=source_data_eval['attention_mask'].to(self.device))
                # source_class_loss = self.loss_calculator(class_output, source_label_eval['class_label'].to(self.device))
                #
                # self.eval_class_loss += source_class_loss.item() / len(eval_loader)
                

            # TODO: EarlyStop 到底用啥 loss 啊? 现在用的是验证集上的全部损失, 尝试 1 + 1 + 2
            '''
            w domain
            '''
            # eval_loss = self.eval_doamin_loss # self.eval_class_loss + self.eval_doamin_loss
            eval_loss = self.eval_doamin_loss + self.eval_class_loss  # self.eval_class_loss + self.eval_doamin_loss
            self.eval_loss_list.append(eval_loss)
            self.eval_class_loss_list.append(self.eval_class_loss)
            self.eval_doamin_loss_list.append(self.eval_doamin_loss)

            '''
            wo domain
            '''
            # self.eval_class_loss_list.append(self.eval_class_loss)
            # eval_loss = self.eval_class_loss
            # self.eval_loss_list.append(eval_loss)

        # '''
        # Test model
        # '''
        # self.model.eval()
        #
        # class_predict_list = list()
        # domain_predict_list = list()
        # class_gt_list = list()
        # domain_gt_list = list()
        #
        # with torch.no_grad():
        #     for batch in test_loader:
        #         target_data_test = batch[0]
        #         target_label_test = batch[1]
        #
        #         class_output, domain_output = self.model(input_ids=target_data_test['input_ids'].to(self.device),
        #                                                  attention_mask=target_data_test['attention_mask'].to(self.device))
        #
        #         label_predict = class_output.ge(0.5).int().cpu().detach().numpy()
        #         domain_predict = domain_output.ge(0.5).int().cpu().detach().numpy()
        #         class_predict_list.append(label_predict)
        #         domain_predict_list.append(domain_predict)
        #
        #         class_gt_list.append(target_label_test['class_label'])
        #         domain_gt_list.append(target_label_test['domain_label'])
        #
        # class_predict_array = np.concatenate(class_predict_list, axis=0)
        # domain_predict_array = np.concatenate(domain_predict_list, axis=0)
        # class_gt_array = np.concatenate(class_gt_list, axis=0)
        # domain_gt_array = np.concatenate(domain_gt_list, axis=0)
        #
        # # TODO: 汤哥，你自己改吧，我实在是没办法了，这一部分的测试效果不打印出来，最后直接将列表保存画图，或者是直接用 TensorBoard 做可视化
        # self.class_accuracy = accuracy_score(class_gt_array, class_predict_array)
        # # self.class_precision = precision_score(class_predict_array, class_gt_array)
        # # self.class_recall = recall_score(class_predict_array, class_gt_array)
        # self.class_f1_score = f1_score(class_gt_array, class_predict_array, average='macro')
        #
        # # self.class_accuracy_list.append(self.class_accuracy)
        # # self.class_precision_list.append(self.class_precision)
        # # self.class_recall_list.append(self.class_recall)
        # # self.class_f1_score_list.append(self.class_f1_score)
        #
        # self.domain_accuracy = accuracy_score(domain_gt_array, domain_predict_array)
        # # self.domain_precision = precision_score(domain_predict_array, domain_gt_array)
        # # self.domain_recall = recall_score(domain_predict_array, domain_gt_array)
        # self.domain_f1_score = f1_score(domain_gt_array, domain_predict_array, average='macro')
        #
        # # self.domain_accuracy_list.append(self.domain_accuracy)
        # # self.domain_precision_list.append(self.domain_precision)
        # # self.domain_recall_list.append(self.domain_recall)
        # # self.domain_f1_score_list.append(self.domain_f1_score)
        #
        # # return self.class_f1_score

        return eval_loss

    
    def train(self, options: dict,
              train_loader: LogDatasetDomainAdaptation_Train, 
              eval_loader: LogDatasetDomainAdaptation_Eval, 
              test_loader: LogDatasetDomainAdaptation_Test):

        epochs = options['max_epoch']

        with tqdm(total=epochs, desc='Model training') as pbar:
            for epoch in range(1, epochs+1):
                eval_loss = self._train_one_epoch(train_loader=train_loader, eval_loader=eval_loader,
                                                  test_loader=test_loader, epoch=epoch, epochs=epochs)
                self.early_stop(val_loss=eval_loss, model=self.model, 
                                path=os.path.join(self.model_save_path, '{}_{}.pt'.format(self.model.name(), epoch)),
                                filepath=self.model_save_path)

                '''
                w domain
                '''
                # tqdm.write('epoch: {:5} | lr: {:.16f} | train class loss: {:.16f} | train domain loss: {:.16f} | eval class loss: {:.16f} | eval domain loss: {:.16f} | test class f1 score: {:.16f}'. \
                #             format(epoch, self.scheduler.get_last_lr()[0]/0.95, self.train_class_loss, self.train_doamin_loss,
                #                    self.eval_class_loss, self.eval_doamin_loss, self.class_f1_score))
                tqdm.write('epoch: {:5} | lr: {:.16f} | train class loss: {:.16f} | train domain loss: {:.16f} | eval class loss: {:.16f} | eval domain loss: {:.16f}'. \
                            format(epoch, self.scheduler.get_last_lr()[0]/0.95, self.train_class_loss, self.train_doamin_loss,
                                   self.eval_class_loss, self.eval_doamin_loss))

                '''
                wo domain
                '''
                # tqdm.write('epoch: {:4} | lr: {:.16f} | train class loss: {:.16f} | eval class loss: {:.16f}'. \
                #             format(epoch, self.scheduler.get_last_lr()[0]/0.95, self.train_class_loss, self.eval_class_loss, ))

                pbar.update(1)

                if self.early_stop.early_stop:
                    print('\nEarly stopping!')

                    break

        """
        w domain
        """
        loss_dict = {'train_class_loss_array': np.array(self.train_class_loss_list),
                     'train_domain_loss_array': np.array(self.train_doamin_loss_list),
                     'eval_class_loss_array': np.array(self.eval_class_loss_list),
                     'eval_domain_loss_array': np.array(self.eval_doamin_loss_list),
                     'eval_loss_array': np.array(self.eval_loss_list),

                     'test_class_accuracy_array': np.array(self.class_accuracy_list),
                     'test_class_precision_array': np.array(self.class_precision_list),
                     'test_class_recall_array': np.array(self.class_recall_list),
                     'test_class_f1_score_array': np.array(self.class_f1_score_list),

                     'test_domain_accuracy_array': np.array(self.domain_accuracy_list),
                     'test_domain_precision_array': np.array(self.domain_precision_list),
                     'test_domain_recall_array': np.array(self.domain_recall_list),
                     'test_domain_f1_score_array': np.array(self.domain_f1_score_list)}

        joblib.dump(loss_dict, os.path.join(self.loss_save_path, 'loss.dict'))

    def test(self, weight_file_path: str, test_loader: LogDatasetDomainAdaptation_Test):

        for name in os.listdir(weight_file_path):
            if name[-2:] == 'pt':
                file_path = os.path.join(weight_file_path, name)

        self.model.load_state_dict(torch.load(file_path))
        self.model.to(self.device)
        self.model.eval()

        class_predict_list = list()
        domain_predict_list = list()
        class_gt_list = list()
        domain_gt_list = list()

        with torch.no_grad():
            for batch in test_loader:
                target_data_test = batch[0]
                target_label_test = batch[1]

                '''
                w domain
                '''
                class_output, domain_output = self.model(input_ids=target_data_test['input_ids'].to(self.device),
                                                         attention_mask=target_data_test['attention_mask'].to(self.device),
                                                         token_type_ids=None,
                                                         alpha=0)

                label_predict = class_output.ge(0.5).int().cpu().detach().numpy()
                domain_predict = domain_output.ge(0.5).int().cpu().detach().numpy()
                class_predict_list.append(label_predict)
                domain_predict_list.append(domain_predict)

                class_gt_list.append(target_label_test['class_label'])
                domain_gt_list.append(target_label_test['domain_label'])

            class_predict_array = np.concatenate(class_predict_list, axis=0)
            domain_predict_array = np.concatenate(domain_predict_list, axis=0)
            class_gt_array = np.concatenate(class_gt_list, axis=0)
            domain_gt_array = np.concatenate(domain_gt_list, axis=0)

            self.class_accuracy = accuracy_score(class_predict_array, class_gt_array)
            self.class_classification_report = classification_report(class_gt_array, class_predict_array, )

            print('class classification accuracy: {}'.format(self.class_accuracy))
            print(self.class_classification_report)

            self.domain_accuracy = accuracy_score(domain_predict_array, domain_gt_array)
            self.domain_classification_report = classification_report(domain_gt_array, domain_predict_array, )

            print('domain classification accuracy: {}'.format(self.domain_accuracy))
            print(self.domain_classification_report)

                # '''
                # wo domain
                # '''
                # class_output = self.model(input_ids=target_data_test['input_ids'].to(self.device),
                #                           attention_mask=target_data_test['attention_mask'].to(self.device))
                #
                # label_predict = class_output.ge(0.5).int().cpu().detach().numpy()
                # class_predict_list.append(label_predict)
                #
                # class_gt_list.append(target_label_test['class_label'])

        # class_predict_array = np.concatenate(class_predict_list, axis=0)
        # class_gt_array = np.concatenate(class_gt_list, axis=0)
        #
        # self.class_accuracy = accuracy_score(class_predict_array, class_gt_array)
        # self.class_classification_report = classification_report(class_gt_array, class_predict_array, )
        #
        # print('class classification accuracy: {}'.format(self.class_accuracy))
        # print(self.class_classification_report)
