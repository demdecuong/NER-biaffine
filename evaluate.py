import os
import json
import pytz
import torch
import datetime
import fasttext

from os import path
from config import Config
from transformers import AutoTokenizer

from dataset import MyDataSet
from trainer import Trainer
from model import Model
from utils import load_model
from config import get_config


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if args.use_fasttext:
        fasttext_model = fasttext.load_model(args.fasttext_path)
    else:
        fasttext_model = None
    print('--------------------- MODEL SETTING UP ---------------------')
    print(f'Loading model from checkpoint {args.load_ckpt}')
    model = Model(args)
    model.load_state_dict(torch.load(args.load_ckpt, map_location=torch.device('cpu')))


    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("#Params = ", params_num)

    print('--------------------- DATALOADER ---------------------')
    dev_data = MyDataSet(
        name='dev',
        path=args.dev_data,
        args=args,
        tokenizer=tokenizer,
        fasttext_model=fasttext_model)

    test_data = MyDataSet(
        name='test',
        path=args.test_data,
        args=args,
        tokenizer=tokenizer,
        fasttext_model=fasttext_model)

    human_test_data = MyDataSet(
        name='human_test',
        path=args.human_test_data,
        args=args,
        tokenizer=tokenizer,
        fasttext_model=fasttext_model)

    trainer = Trainer(args=args,
                      model=model,
                      train_data=None,
                      dev_data=dev_data,
                      test_data=test_data,
                      human_test_data=human_test_data)

    dev_prec, dev_recal, dev_f1 = trainer.eval('dev')
    test_prec, test_recall, test_f1 = trainer.eval('test')
    htest_prec, htest_recall, htest_f1 = trainer.eval('human_test')


if __name__ == "__main__":
    args = get_config()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using Device : {args.device}')

    if torch.cuda.is_available():
        print(f"GPU device : {torch.cuda.get_device_name(0)}")

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    print('--------------------- EVALUATING ---------------------')
    main(args)
