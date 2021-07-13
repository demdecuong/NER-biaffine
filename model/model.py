from torch import nn

from model.layer import WordRepresentation, FeedforwardLayer, BiaffineLayer
from transformers import AutoConfig

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        config = AutoConfig.from_pretrained(args.model_name_or_path)
        
        self.use_pos = args.use_pos

        self.num_labels = args.num_labels
        
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
        
        self.word_rep = WordRepresentation(args)
        self.bilstm = nn.LSTM(input_size=self.lstm_input_size,
                            hidden_size=args.hidden_dim // 2,
                            num_layers=args.rnn_num_layers,
                            bidirectional=True,
                            batch_first=True)
        self.feedStart = FeedforwardLayer(d_in=args.hidden_dim, d_hid=args.hidden_dim_ffw)
        self.feedEnd = FeedforwardLayer(d_in=args.hidden_dim, d_hid=args.hidden_dim_ffw)
        self.biaffine = BiaffineLayer(inSize1=args.hidden_dim, inSize2=args.hidden_dim, classSize=self.num_labels)

    def forward(self, input_ids=None, char_ids=None, fasttext_embs=None, first_subword=None, attention_mask=None, pos_ids=None,train=False):

        word_features = self.word_rep(input_ids=input_ids, 
                                    attention_mask=attention_mask,
                                    first_subword=first_subword,
                                    char_ids=char_ids,
                                    pos_ids=pos_ids,
                                    train=train)

        x, _ = self.bilstm(word_features)

        start = self.feedStart(x)
        end = self.feedEnd(x)

        score = self.biaffine(start, end)

        return score