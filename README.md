# CharEmbedding + BERT + BiLSTM + Biaffine for Vietnamese person name recognition

## Installation
`pip install requirement.txt`

### Model
There are 3 main options for using BERT which is `args.model_name_or_path` in this pipeline 
- `use_bert=True`: use embedding + transformer layers of BERT  
- `bert_embed_only=True` : only using embedding in BERT
- `use_bert=False`, `bert_embed_only=False`, `num_layer_bert=2` : use embedding + 2 layers in BERT

### Training model 

1. Train normal
```
python main.py --aug_offline False --aug_online False --ckpt_dir ckpt
```
2. Train augmentation
```
python main.py --aug_offline True \
    --aug_lastname 0.2 \
    --aug_lowercase 0.2 \     
    --aug_acent 0.1
    --aug_remove 0.15 
    --ckpt_dir ckpt_aug_offline
```
3. Train augmentation online
```
python main.py --aug_online True  \
    --use_aug_every 3
    --aug_lastname 0.2 \
    --aug_lowercase 0.2 \     
    --aug_acent 0.1 
    --aug_remove 0.15 
    --ckpt_dir ckpt_aug_online
    
```
