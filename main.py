from os import path
from config import Config

from transformers import AutoTokenizer
import torch

from dataset import MyDataSet
from trainer import Trainer
from model import Model
from utils import load_model
from config import get_config
import fasttext

import json
import os

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if args.use_fasttext:
        fasttext_model = fasttext.load_model(args.fasttext_path)
    else:
        fasttext_model = None
    print('--------------------- MODEL SETTING UP ---------------------')
    if args.use_pretrained:
        print(f'Use pretrained model from checkpoint {args.checkpoint}')
        model = Model(args)
        model.load_state_dict(torch.load(
            args.checkpoint, map_location=torch.device('cpu')))
        print(f'Model is loaded weights from checkpoint')
    else:
        print(f'Initialize Name Entity Recognition as Dependence Parsing .  ..  ...')
        model = Model(args)

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

    train_data = MyDataSet(
        name='train',
        path=args.train_data,
        args=args,
        tokenizer=tokenizer,
        fasttext_model=fasttext_model,
        use_aug=args.aug_offline)

    trainer = Trainer(args=args,
                      model=model,
                      train_data=train_data,
                      dev_data=dev_data,
                      test_data=test_data)
    f1_pre = 0
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    print('--------------------- TRAINING ---------------------')

    for i in range(1, args.iteration + 1):
        print(f'Training model on iteration {i} ...')
    
        for e in range(1, args.num_epochs + 1):
            print(f'Training model on epoch {e} of iteration {i}')

            if args.do_train:
                trainer.train()
                f1_score = trainer.eval('test', f1_pre)
                if f1_score > f1_pre:
                    # trainer.save_model(f1_score)
                    torch.save(trainer.model.state_dict(),
                               f'./{args.ckpt_dir}/checkpoint_{f1_score}.pth')
                    f1_pre = f1_score
                else:
                    f1_pre = f1_pre
            if args.do_eval:
                print('--------------------- TESTING ---------------------')

                test_f1 = trainer.eval('test', f1_pre)
                eval_f1 = trainer.eval('dev', f1_pre)
                torch.save(model.state_dict(), 'checkpoint.pth')


def main_aug_online(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if args.use_fasttext:
        fasttext_model = fasttext.load_model(args.fasttext_path)
    else:
        fasttext_model = None
    print('--------------------- MODEL SETTING UP ---------------------')
    if args.use_pretrained:
        print(f'Use pretrained model from checkpoint {args.checkpoint}')
        model = Model(args)
        model.load_state_dict(torch.load(
            args.checkpoint, map_location=torch.device('cpu')))
        print(f'Model is loaded weights from checkpoint')
    else:
        print(f'Initialize Name Entity Recognition as Dependence Parsing .  ..  ...')
        model = Model(args)

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

    f1_pre = 0
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for i in range(1, args.iteration + 1):
        print('--------------------- TRAINING ---------------------')
        use_aug_iter = False
        if i % args.use_aug_every == 0:
            use_aug_iter = True
            print(f'Training model on iteration {i} [WITH] augmentation ...')
        else:
            print(f'Training model on iteration {i} without augmentation ...')

        train_data = MyDataSet(
            name='train',
            path=args.train_data,
            args=args,
            tokenizer=tokenizer,
            fasttext_model=fasttext_model,
            use_aug=use_aug_iter)
        trainer = Trainer(
            args=args,
            model=model,
            train_data=train_data,
            dev_data=dev_data,
            test_data=test_data
            )
        for e in range(1, args.num_epochs + 1):
            print(f'Training model on epoch {e} of iteration {i}')
            if args.do_train:
                trainer.train()
                f1_score = trainer.eval('test', f1_pre)
                if f1_score > f1_pre:
                    # trainer.save_model(f1_score)
                    torch.save(trainer.model.state_dict(),
                               f'./{args.ckpt_dir}/checkpoint_{f1_score}.pth')
                    f1_pre = f1_score
                else:
                    f1_pre = f1_pre
            if args.do_eval:
                print('--------------------- TESTING ---------------------')

                test_f1 = trainer.eval('test', f1_pre)
                eval_f1 = trainer.eval('dev', f1_pre)
                # torch.save(model.state_dict(), 'checkpoint.pth')


if __name__ == "__main__":
    args = get_config()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
        
    if args.aug_online:
        print(
            f'[INFO] Train Augmentation Online with epoch = {args.num_epochs} for each iteration = {args.iteration}')
        main_aug_online(args)
    else:
        print('[INFO] Train Normal')
        main(args)
