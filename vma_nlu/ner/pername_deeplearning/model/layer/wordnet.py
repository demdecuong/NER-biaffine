import torch
from vma_nlu.ner.pername_deeplearning.model.layer.charcnn import CharCNN
from vma_nlu.ner.pername_deeplearning.model.layer.featureEmbed import FeatureEmbedding

import torch
from torch import nn
from transformers import AutoModel

from vma_nlu.ner.pername_deeplearning.utils import get_extended_attention_mask

class WordRepresentation(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.bert_embed_only = args.bert_embed_only
        self.num_layer_bert = args.num_layer_bert
        self.eval_num_layer_bert = args.eval_num_layer_bert

        assert self.eval_num_layer_bert >= self.num_layer_bert

        self.use_bert = args.use_bert

        if self.use_bert:
            self.bert = AutoModel.from_pretrained(args.model_name_or_path)
        elif self.bert_embed_only:
            self.bert_embed = AutoModel.from_pretrained(args.model_name_or_path).base_model.embeddings
        else:
            print(f'Using {self.num_layer_bert} layers and embedding in {args.model_name_or_path}')
            self.bert_embed = AutoModel.from_pretrained(args.model_name_or_path).base_model.embeddings
            self.bert_layers = nn.ModuleList([AutoModel.from_pretrained(args.model_name_or_path).base_model.encoder.layer[i].to(args.device) for i in range(args.num_layer_bert)])

        self.use_char = args.use_char
        if self.use_char:
            self.char_feature = CharCNN(hidden_dim=args.char_hidden_dim,
                                        vocab_size=args.char_vocab_size, 
                                        embedding_dim=args.char_embedding_dim)

        self.use_pos = args.use_pos
        if self.use_pos:
            self.pos_feature = FeatureEmbedding(vocab_size=args.pos_vocab_size, 
                                                embedding_size=args.feature_embed_dim)


    def forward(self, input_ids, attention_mask, first_subword, char_ids, pos_ids, train=False):
        if self.use_bert:
            bert_output = self.bert(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            bert_features = []
            if train != False:
                for i in range(-1, -(self.num_layer_bert + 1), -1):
                    bert_features.append(bert_output[2][i])
            else:
                for i in range(-1, -(self.eval_num_layer_bert + 1), -1):
                    bert_features.append(bert_output[2][i])
        
            bert_features = torch.cat(bert_features, dim=-1)

            bert_features = torch.cat([torch.index_select(bert_features[i], 0, first_subword[i]).unsqueeze(0) for i in range(bert_features.size(0))], dim=0)
        elif self.bert_embed_only:
            bert_features = self.bert_embed(input_ids)
        else:
            bert_output = self.bert_embed(input_ids)
            attention_mask = get_extended_attention_mask(attention_mask,input_ids.shape,input_ids.device)
            bert_features = []
            for layer in self.bert_layers:
                residual = bert_output
                bert_output = layer(bert_output, attention_mask=attention_mask)[0]
                bert_output += residual
                bert_features.append(bert_output)

            bert_features = torch.cat(bert_features, dim=-1)
            bert_features = torch.cat([torch.index_select(bert_features[i], 0, first_subword[i]).unsqueeze(0) for i in range(bert_features.size(0))], dim=0)
        
        word_features = []
        if self.use_char:
            char_features = self.char_feature(char_ids)
            word_features.append(char_features)
        
        if self.use_pos:
            pos_features = self.pos_feature(pos_ids)
            word_features.append(pos_features)

        word_features = torch.cat(word_features, dim=-1)

        final_feature = torch.cat((bert_features, word_features), dim=-1)
        return final_feature