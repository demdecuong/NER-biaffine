# Inference via vma_nlu

## Installation

```
pip install -r requirements.txt
```

## Usage
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


Please change config/checkpoint dir in `vma_nlu/ner/pername_deeplearning/__init__.py`
