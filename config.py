import argparse

class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)

def get_config(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_name_or_path', type=str, default='vinai/phobert-base')
    parser.add_argument('--pos_tag_set_path', default=None)
    parser.add_argument('--label_set_path', type=str, default='./data/person_name/label_set.txt')
    parser.add_argument('--char_vocab_path', type=str, default='./data/person_name/charindex.json')
    parser.add_argument('--fasttext_path', type=str, default= './data/cc.vi.300.bin')
    parser.add_argument('--save_folder', type=str, default='final_checkpoint')
    parser.add_argument('--ckpt_dir', type=str, default='save_checkpoint')
    parser.add_argument('--use_pretrained', type=bool, default=False)

    # Augmentation
    parser.add_argument('--aug_lastname', type=float, default=0) 
    parser.add_argument('--aug_lowercase', type=float, default=0)     
    parser.add_argument('--aug_acent', type=float, default=0) 
    parser.add_argument('--aug_replace', type=float, default=0) 
    parser.add_argument('--aug_insert', type=float, default=0) 
    parser.add_argument('--aug_remove', type=float, default=0) 
    parser.add_argument('--aug_online', type=bool, default=False) 
    parser.add_argument('--use_aug_every', type=int, default=3) 
    parser.add_argument('--aug_offline', type=bool, default=False) 

    # optional features
    parser.add_argument('--use_bert', type=bool, default=False)
    parser.add_argument('--bert_embed_only', type=bool, default=False) 
    parser.add_argument('--use_pos', type=bool, default=False)
    parser.add_argument('--use_char', type=bool, default=True)
    parser.add_argument('--use_fasttext', type=bool, default=False)

    # Model
    parser.add_argument('--num_layer_bert', type=int, default=2)
    parser.add_argument('--eval_num_layer_bert', type=int, default=2)
    parser.add_argument('--char_hidden_dim', type=int, default=128)
    parser.add_argument('--char_embedding_dim', type=int, default=100)
    parser.add_argument('--feature_embed_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=728)
    parser.add_argument('--hidden_dim_ffw', type=int, default=300)
    parser.add_argument('--char_vocab_size', type=int, default=108)
    parser.add_argument('--pos_vocab_size', type=int, default=23)
    parser.add_argument('--rnn_num_layers', type=int, default=3)
    parser.add_argument('--max_char_len', type=int, default=20)
    parser.add_argument('--max_seq_len', type=int, default=100) # vinai : 190 # vlsp2016: 160 # custom

    parser.add_argument('--num_labels', type=int, default=2) 
    
    # train
    parser.add_argument('--iteration', type=int, default=20) 
    parser.add_argument('--batch_size', type=int, default=80) 
    parser.add_argument('--num_epochs', type=int, default=10) 
    parser.add_argument('--learning_rate', type=float, default=5e-5) 
    parser.add_argument('--adam_epsilon', type=float, default=1e-8) 
    parser.add_argument('--weight_decay', type=float, default=0.01) 
    parser.add_argument('--warmup_steps', type=int, default=0) 
    parser.add_argument('--max_grad_norm', type=int, default=1)
     
    parser.add_argument('--do_train', type=bool, default=True) 
    parser.add_argument('--do_eval', type=bool, default=True) 

    # data
    parser.add_argument('--train_data', type=str, default='./data/person_name/train.json') 
    parser.add_argument('--dev_data', type=str, default='./data/person_name/test_60.json') 
    parser.add_argument('--test_data', type=str, default='./data/person_name/data_synth.json') 

    kwargs = parser.parse_args()
    kwargs.checkpoint = f'./{kwargs.save_folder}/checkpoint.pth'
    kwargs = vars(kwargs)
    print('--------------------- CONFIG ---------------------')
    print(kwargs)
    return Config(**kwargs)