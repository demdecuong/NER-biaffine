import os
import re
import random
import pandas as pd
import json

def save_to_json(data, out_path):
    print(f'Saving to {out_path} ... ')
    with open(out_path,'w',encoding='utf-8') as f:
        for sample in data:
            f.write(str(json.dumps(sample,ensure_ascii=False)) + '\n')
    
def read(path):
    data = []
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def remove_label(data,remove_label=['DATE','[]']):
    final_data = []
    for sample in data:
        label = []
        for lab in sample['label']:
            if lab[-1] not in remove_label:
                label.append(lab) 
            else:
                print(label)
        
        obj = {
            'sentence' : sample['sentence'],
            'label' : label
        }
        final_data.append(obj)
    return final_data

if __name__ == '__main__':
    dev_data = read('dev/seq_40/dev_40.json')
    dev_data = remove_label(dev_data)
    save_to_json(dev_data,'dev_aug_40.json')