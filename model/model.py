from torch import nn

from model.layer import WordRepresentation, FeedforwardLayer, BiaffineLayer
from model.layer.tcn import  TemporalConvNet
from transformers import AutoConfig


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        config = AutoConfig.from_pretrained(args.model_name_or_path)
        
        self.use_pos = args.use_pos

        self.num_labels = args.num_labels

        self.use_tcn = args.use_tcn
        
        if args.bert_embed_only:
            self.lstm_input_size = config.hidden_size
        else:
            self.lstm_input_size = args.num_layer_bert * config.hidden_size

        if args.use_char:
            self.lstm_input_size = self.lstm_input_size + 2 * args.char_hidden_dim
        
        if args.use_pos:
            self.lstm_input_size = self.lstm_input_size + args.feature_embed_dim

        if args.use_fasttext:
            self.lstm_input_size = self.lstm_input_size + 300
        
        if args.use_tcn:
            num_channels = [args.nhid] * (args.levels - 1) + [self.lstm_input_size]
            self.tcn = TemporalConvNet(self.lstm_input_size, num_channels, args.kernel_size, dropout=args.dropout)
            self.embed_drop = nn.Dropout(0.1)
            self.decoder = nn.Linear(num_channels[-1], args.hidden_dim)
        else:
            self.bilstm = nn.GRU(input_size=self.lstm_input_size,
                    hidden_size=args.hidden_dim // 2,
                    num_layers=args.rnn_num_layers,
                    bidirectional=True,
                    batch_first=True)

        self.word_rep = WordRepresentation(args)

        self.feedStart = FeedforwardLayer(d_in=args.hidden_dim, d_hid=args.hidden_dim_ffw, dropout=args.dropout)
        self.feedEnd = FeedforwardLayer(d_in=args.hidden_dim, d_hid=args.hidden_dim_ffw, dropout=args.dropout)

        self.biaffine = BiaffineLayer(inSize1=args.hidden_dim, inSize2=args.hidden_dim, classSize=self.num_labels)

    def forward(self, input_ids=None, char_ids=None, fasttext_embs=None, first_subword=None, attention_mask=None, pos_ids=None,train=False):

        word_features = self.word_rep(input_ids=input_ids, 
                                    attention_mask=attention_mask,
                                    first_subword=first_subword,
                                    char_ids=char_ids,
                                    pos_ids=pos_ids,
                                    train=train)
        if self.use_tcn:
            emb = self.embed_drop(word_features)
            tcn_out = self.tcn(emb.transpose(1, 2)).transpose(1, 2)
            x = self.decoder(tcn_out)
        else:
            x, _ = self.bilstm(word_features) # b x len x d

        start = self.feedStart(x)
        end = self.feedEnd(x)

        score = self.biaffine(start, end)

        return score