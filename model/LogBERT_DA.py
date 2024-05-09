import torch.nn as nn
import torch
from torch.autograd import Function
from transformers import BertConfig, BertModel


class GRL(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
    

class LogBERT_DA(nn.Module):

    '''
    FIXME: 新增 options 参数
    * class_pred_dim: dimension of class predict feedforward network
    * domain_pred_dim: dimension of domain predict feedforward network
    '''
    def __init__(self, class_pred_dim: int = 64,
                 domain_pred_dim: int = 64,
                 class_pred_dropout_ratio: float = 0.1,
                 domain_pred_dropout_ratio: float = 0.1,
                 grl_alpha: float = 0.5,
                 bert_config_file_path: str = './bert_base_checkpoint'
                 ) -> None:
        super(LogBERT_DA, self).__init__()

        print('loading bert checkpoint...')
        feature_extractor_config = BertConfig.from_pretrained(bert_config_file_path)
        self.feature_extractor = BertModel.from_pretrained(bert_config_file_path, config=feature_extractor_config)
        print('loading finish :D\n')

        # for param in self.feature_extractor.parameters():
        #     param.requires_grad_(False)

        self.class_predictor = nn.Sequential(
            # nn.Dropout(p=class_pred_dropout_ratio),
            nn.Linear(self.feature_extractor.config.hidden_size, class_pred_dim),
            nn.GELU(),
            nn.Linear(class_pred_dim, 1),
            # nn.Sigmoid(),
        )

        self.domain_classifier = nn.Sequential(
            # nn.Dropout(p=domain_pred_dropout_ratio),
            nn.Linear(self.feature_extractor.config.hidden_size, domain_pred_dim),
            nn.GELU(),
            nn.Linear(domain_pred_dim, 1),
            # nn.Sigmoid(), 
        )

        self.sigmoid = nn.Sigmoid()

        self.alpha = grl_alpha

    def forward(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor = None, ):
        
        bert_output = self.feature_extractor(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # bert_output = torch.mean(bert_output['last_hidden_state'], dim=1)
        bert_output = bert_output['last_hidden_state'][:, 0, :]

        class_output = self.class_predictor(bert_output)
        class_output = self.sigmoid(class_output)

        reverse_bert_output = GRL.apply(bert_output, self.alpha)
        domain_output = self.domain_classifier(reverse_bert_output)
        domain_output = self.sigmoid(domain_output)

        return class_output, domain_output
    
    def name(self, ):
        return self.__class__.__name__
