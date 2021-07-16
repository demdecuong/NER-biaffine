# Train/Test data

## Folder structure: 
- `aug_data` : synthesize data from template
- `person_name` : all data related to person name
- `format` : format of standard data config

## Train data
- `person_name/train.json` : trainset is collected from vinai-covid19, vlsp16 and vlsp18 
- `person_name/augment_data.json` : augment lowercase/lastname_only/remove_accent/remove_name from `train.json`
- `person_name/train_synth.json` : trainset from synthesize data
- `person_name/train_aug_synth.json` : trainset + synthesize data

## Test data
- `person_name/dev_60.json` : testset is collected from vinai-covid19, vlsp16 and vlsp18 
- `person_name/data_synth.json` : testset is synthesize data
- `person_name/human_test.json` : testset from human annotation


## Usage
1. Convert csv testset/synthesize data into trainable/testable format
```
python format_testset.py
```
