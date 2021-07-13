import json
import os
import random
import re

import torch
from torch.utils.data import Dataset

import numpy as np


class InputSample(object):
    def __init__(self, args, path='./data/train.json', max_char_len=None, pos_tag_set_path=None, augment=False):

        self.max_char_len = max_char_len
        self.list_samples = []
        # sample: {'sentence': 'tôi yêu em tại HCM', 'label': [[4,5, ORG]], 'idx':1}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                self.list_samples.append(json.loads(line))
        if pos_tag_set_path:
            self.use_pos = True
        else:
            self.use_pos = False

        # Augmentation
        if augment:
            self.aug_lastname = args.aug_lastname
            self.aug_lowercase = args.aug_lowercase
            self.aug_acent = args.aug_acent
            self.aug_replace = args.aug_replace
            self.aug_insert = args.aug_insert
            self.aug_remove = args.aug_remove
            print('# train data samples:', len(self.list_samples))

            self.list_aug_samples = self.augment_data(self.list_samples)

            self.list_samples.extend(self.list_aug_samples)
            with open('./data/person_name/augment_data.json', 'w', encoding='utf-8') as f:
                for line in self.list_samples:
                    f.write(str(json.dumps(line,ensure_ascii=False)) + '\n')
            self.list_samples = []
            with open('./data/person_name/augment_data.json', 'r', encoding='utf-8') as f:
                for line in f:
                    self.list_samples.append(json.loads(line))
            print('# train data samples after augment:', len(self.list_samples))

    def get_sample(self):
        for idx, sample in enumerate(self.list_samples):
            sentence = sample['sentence'].split()
            sample['sentence'] = sentence
            char_seq = []
            for word in sentence:
                character = self.get_character(word, self.max_char_len)
                char_seq.append(character)
            sample['char-sequence'] = char_seq
            if self.use_pos:
                pos_label = sample['pos-tag'].split()
                sample['pos-tag'] = pos_label
            sample['id'] = idx
            self.list_samples[idx] = sample
        return self.list_samples

    def get_character(self, word, max_char_len):
        word_seq = []
        for i in range(max_char_len):  # 20
            try:
                char = word[i]
            except:
                char = 'PAD'
            word_seq.append(char)
        return word_seq

    def augment_data(self, list_samples):
        aug_data = []
        # augment lastname
        for sample in list_samples:
            lucky_num = random.random()
            if lucky_num <= self.aug_lastname:
                aug_sample = self.augment_lastname(sample)
                if aug_sample not in aug_data:
                    aug_data.append(aug_sample)
                assert type(aug_sample['sentence']) == str

        # augment lower case
        for sample in list_samples:
            lucky_num = random.random()
            if lucky_num <= self.aug_lowercase:
                aug_sample = self.augment_lowercase(sample)
                if aug_sample not in aug_data:
                    aug_data.append(aug_sample)
                assert type(aug_sample['sentence']) == str
        # augment acent
        for sample in list_samples:
            lucky_num = random.random()
            if lucky_num <= self.aug_acent:
                aug_sample = self.augment_acent(sample)
                if aug_sample not in aug_data:
                    aug_data.append(aug_sample)
                assert type(aug_sample['sentence']) == str

        # augment insert
        for sample in list_samples:
            lucky_num = random.random()
            if lucky_num <= self.aug_acent:
                aug_sample = self.augment_insert(sample)
                if aug_sample not in aug_data:
                    aug_data.append(aug_sample)
                assert type(aug_sample['sentence']) == str

        # augment remove
        for sample in list_samples:
            lucky_num = random.random()
            if lucky_num <= self.aug_acent:
                if len(sample['sentence'].split()) > 90:
                    continue
                aug_sample = self.augment_remove(sample)
                if aug_sample not in aug_data:
                    aug_data.append(aug_sample)
                assert type(aug_sample['sentence']) == str

        return aug_data

    def augment_lastname(self, sample):
        sent = sample['sentence'].split()
        label_s, label_e = sample['label'][0][0], sample['label'][0][1]
        remove_idx = label_s
        while label_s < label_e:
            sent.remove(sent[remove_idx])
            label_s += 1
        sample['sentence'] = ' '.join(sent)
        sample['label'][0] = [remove_idx, remove_idx, 'PERSON']
        return sample

    def augment_lowercase(self, sample):
        sent = sample['sentence'].split()
        label_s, label_e = sample['label'][0][0], sample['label'][0][1]
        # sent[label_s:label_e + 1] = [token.lower()
        #                              for token in sent[label_s:label_e + 1]]
        sent = [token.lower() for token in sent]
        sample['sentence'] = ' '.join(sent)
        sample['label'][0] = [label_s, label_e, 'PERSON']
        return sample

    def augment_acent(self, sample):
        sent = sample['sentence'].split()
        label_s, label_e = sample['label'][0][0], sample['label'][0][1]
        # sent[label_s:label_e +
        #      1] = [remove_accent_vietnamese(token) for token in sent[label_s:label_e + 1]]
        sent = [remove_accent_vietnamese(token) for token in sent]
        sample['sentence'] = ' '.join(sent)
        sample['label'][0] = [label_s, label_e, 'PERSON']
        return sample

    def augment_replace(self, sample):
        pass

    def augment_insert(self, sample):
        sent = sample['sentence'].split()
        label_s, label_e = sample['label'][0][0], sample['label'][0][1]
        insert_token = random.choice(sent)
        sent = sent[:label_s] + [insert_token] + sent[label_s:]
        sample['sentence'] = ' '.join(sent)
        sample['label'][0] = [label_s, label_e + 1, 'PERSON']
        return sample

    def augment_remove(self, sample):
        sent = sample['sentence'].split()
        label_s, label_e = sample['label'][0][0], sample['label'][0][1]
        remove_idx = label_s
        try:
            while label_s <= label_e:
                sent.remove(sent[remove_idx])
                label_s += 1
        except:
            sent.remove(sent[len(sent)-1])
        sample['sentence'] = ' '.join(sent)
        if len(sample['label']) > 1:
            sample['label'].pop(0)
        else:
            sample['label'] = []
        return sample


class MyDataSet(Dataset):
    def __init__(self,
                 name,
                 path,
                 args,
                 tokenizer,
                 fasttext_model,
                 use_aug=False) -> None:
        super().__init__()

        self.max_char_len = args.max_char_len
        self.max_seq_len = args.max_seq_len
        self.use_fasttext = args.use_fasttext

        self.tokenizer = tokenizer
        self.fasttext_model = fasttext_model

        self.name = name
        if self.name == 'train' and use_aug == True:
            samples = InputSample(args, path=path, max_char_len=self.max_char_len,
                                  pos_tag_set_path=args.pos_tag_set_path, augment=True).get_sample()
        else:
            samples = InputSample(args, path=path, max_char_len=self.max_char_len,
                                  pos_tag_set_path=args.pos_tag_set_path, augment=False).get_sample()

        with open(args.label_set_path, 'r', encoding='utf-8') as f:
            self.label_set = f.read().splitlines()
        self.label_set = {w: i for i, w in enumerate(self.label_set)}
        print(f'Label set of entities: {self.label_set}')

        # update label

        # add noise

        self.samples = samples

        if args.pos_tag_set_path:
            self.pos_tag = True
            with open(args.pos_tag_set_path, 'r', encoding='utf-8') as f:
                self.pos_tag_set = f.read().splitlines()
            self.pos_tag_set = {w: i for i, w in enumerate(self.pos_tag_set)}
            print(f'Label set of POS: {self.pos_tag_set}')
        else:
            self.pos_tag = False

        with open(args.char_vocab_path, 'r', encoding='utf-8') as f:
            self.char_vocab = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        id_sample = sample['id']
        sentence = sample['sentence']
        char_seq = sample['char-sequence']

        if self.pos_tag:
            pos_tag = sample['pos-tag']
            pos_ids = self.pos_tag2id(pos_tag, self.max_seq_len)

        seq_len = len(sentence)
        seq_len = torch.tensor([seq_len])

        label = sample['label']
        # print(label)
        label = self.span_maxtrix_label(label)

        input_ids, attention_mask, firstSWindices = self.preprocess(
            self.tokenizer, sentence, self.max_seq_len)
        char_ids = self.char2id(
            char_seq=char_seq, max_seq_len=self.max_seq_len)

        if self.use_fasttext:
            fasttext_embed = self.get_fasttext_embedding(
                sentence, self.max_seq_len)

        if not self.pos_tag and not self.use_fasttext:
            return input_ids, attention_mask, firstSWindices, seq_len, char_ids, id_sample, label
        elif not self.pos_tag and self.use_fasttext:
            return input_ids, attention_mask, firstSWindices, seq_len, char_ids, fasttext_embed, id_sample, label
        elif self.pos_tag and not self.use_fasttext:
            return input_ids, attention_mask, firstSWindices, seq_len, char_ids, pos_ids, id_sample, label
        else:
            return input_ids, attention_mask, firstSWindices, seq_len, char_ids, pos_ids, fasttext_embed, id_sample, label

    def pos_tag2id(self, pos_tag, max_seq_len):
        pos_ids = []

        for pos in pos_tag:
            pos_ids.append(self.pos_tag_set.get(pos))
        if len(pos_ids) < max_seq_len:
            pos_ids += [self.pos_tag_set.get('X')] * (max_seq_len-len(pos_ids))
        else:
            pos_ids = pos_ids[:max_seq_len]
        return torch.tensor(pos_ids)

    def span_maxtrix_label(self, label):
        if not label:
            return torch.sparse.FloatTensor(torch.tensor([[0], [0]]), torch.tensor(0), torch.Size([self.max_seq_len, self.max_seq_len])).to_dense()
        start, end, ent = [], [], []
        for lb in label:
            start.append(lb[0])
            end.append(lb[1])
            ent.append(self.label_set[lb[2]])
        label = torch.sparse.FloatTensor(torch.tensor([start, end]), torch.tensor(
            ent), torch.Size([self.max_seq_len, self.max_seq_len])).to_dense()
        return label

    def preprocess(self, tokenizer, sentence, max_seq_len, mask_padding_with_zero=True):  # sentence_embedding

        input_ids = [tokenizer.cls_token_id]
        firstSWindices = [len(input_ids)]

        for w in sentence:
            word_token = tokenizer.encode(w)
            input_ids += word_token[1: (len(word_token) - 1)]
            firstSWindices.append(len(input_ids))

        firstSWindices = firstSWindices[: (len(firstSWindices) - 1)]
        input_ids.append(tokenizer.sep_token_id)

        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        if len(input_ids) > max_seq_len:
            input_ids = input_ids[:max_seq_len]
            attention_mask = attention_mask[:max_seq_len]
            firstSWindices = firstSWindices[:max_seq_len]
        else:
            attention_mask = attention_mask + \
                [0 if mask_padding_with_zero else 1] * \
                (max_seq_len - len(input_ids))
            input_ids = input_ids + \
                [tokenizer.pad_token_id] * (max_seq_len - len(input_ids))
            firstSWindices = firstSWindices + \
                [0]*(max_seq_len - len(firstSWindices))

        return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(firstSWindices)

    def get_fasttext_embedding(self, sentence, max_seq_len):  # word embedding
        vector = []
        for word in sentence:
            vector.append(self.fasttext_model.get_word_vector(word))
        if len(vector) < max_seq_len:
            vector = vector + \
                [self.fasttext_model.get_word_vector(
                    'pad')] * (max_seq_len-len(vector))
        else:
            vector = vector[:max_seq_len]
        return torch.tensor(vector)

    def char2id(self, char_seq, max_seq_len):  # char - embedding
        char_ids = []
        for word in char_seq:
            word_char_ids = []
            for char in word:
                if char not in self.char_vocab:
                    word_char_ids.append(self.char_vocab.get("UNK"))
                else:
                    word_char_ids.append(self.char_vocab.get(char))
            char_ids.append(word_char_ids)
        if len(char_ids) < max_seq_len:
            char_ids += [[self.char_vocab.get("PAD")] *
                         self.max_char_len]*(max_seq_len - len(char_ids))
        else:
            char_ids = char_ids[:max_seq_len]
        return torch.tensor(char_ids)


def remove_accent_vietnamese(s):
    s = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', s)
    s = re.sub(r'[ÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪ]', 'A', s)
    s = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', s)
    s = re.sub(r'[ÈÉẸẺẼÊỀẾỆỂỄ]', 'E', s)
    s = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', s)
    s = re.sub(r'[ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ]', 'O', s)
    s = re.sub(r'[ìíịỉĩ]', 'i', s)
    s = re.sub(r'[ÌÍỊỈĨ]', 'I', s)
    s = re.sub(r'[ùúụủũưừứựửữ]', 'u', s)
    s = re.sub(r'[ƯỪỨỰỬỮÙÚỤỦŨ]', 'U', s)
    s = re.sub(r'[ỳýỵỷỹ]', 'y', s)
    s = re.sub(r'[ỲÝỴỶỸ]', 'Y', s)
    s = re.sub(r'[Đ]', 'D', s)
    s = re.sub(r'[đ]', 'd', s)
    return s
