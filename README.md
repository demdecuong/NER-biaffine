# CharEmbedding + BERT + BiLSTM + Biaffine for Vietnamese person name recognition
![](asset/overview.png)
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
python main.py --aug_offline False --aug_online False --ckpt_dir ckpt --log_file log_normal.csv
```
2. Train augmentation
```
python main.py --aug_offline True \
--aug_lastname 0.2 --aug_lowercase 0.2 --aug_remove 0.15 \
--ckpt_dir ckpt_aug_offline --log_file log_aug_off.csv
```
3. Train augmentation online
```
python main.py --aug_online True --use_aug_every 3 \
--aug_lastname 0.2 --aug_lowercase 0.2 --aug_remove 0.15 \ 
--ckpt_dir ckpt_aug_online --log_file log_aug_onl.csv
```

4. Train normal with Temporal Convolution Networks (replace BiLSTM with TCN)
```
python main.py --use_tcn True --ckpt_dir ckpt --log_file log_normal.csv
```

5. Finetune trained model
```
python main.py --batch_size 256 --ckpt_dir finetune --learning_rate 2e-4 --log_file log_ft.csv \
--use_pretrained True --load_ckpt 'finetune/checkpoint_0.925.pth'
```

### Evaluate 
1. Evaluate checkpoint
```
python evaluate.py --load_ckpt ckpt/checkpoint_0.959.pth --batch_size 256
```

2. Evaluate baseline model (151M params)
```
python evaluate.py --load_ckpt ckpt_baseline/checkpoint.pth --use_bert True --num_layer_bert 4 --eval_num_layer_bert 4 --human_test_data './data/person_name/human_test.json'
```

### Inference

```
from vma_nlu.ner.pername_deeplearning.inference import Inference
extractor = Inference()

text = 'tôi tên là nam'
res = extractor.inference(text)
print(res)
```
Output would be
```
{'entities': [{'start': 11, 'end': 14, 'entity': 'person_name', 'value': 'nam', 'confidence': 0.9999963045120239}]}
```
Please change config/checkpoint dir in vma_nlu/ner/pername_deeplearning/__init__.py

## Investigate 
```
python investigate.py --load_ckpt ckpt/checkpoint_0.753.pth --max_seq_len 80
```
- stats picture(s) saved at `./asset`
- failure samples saved at `./investigate` 

### TODO  
- Save model per batch
- Complete human testset
- Investigate failure samples

### Result

| Model | #params | vinai-covid/vlsp | human testset |
| --- | --- | --- | --- |
| CharEmbed + BERT + CRF | 151M | **0.997** | 0.716 |
| CharEmbed + BERT(N=2) + CRF | **78M** | 0.945 | **0.755** |