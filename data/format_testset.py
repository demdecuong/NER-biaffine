'''
    Convert csv testset format into trainable format
'''

import pandas as pd

def from_csv_to_json(path,out_path):
    pass

if __name__ == '__main__':
    testset = pd.read_csv("person_name/human_testset.csv")
    testset.fillna("missing")
    trg = testset['input'].tolist()
    name1 = testset['name_1'].tolist()
    name2 = testset['name_2'].tolist()
    print(trg[0])
    print(name1[0])
    print(len(testset))