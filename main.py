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
    if args.use_pretrained:
        print(f'Use pretrained model from checkpoint {args.load_ckpt}')
        model = Model(args)
        model.load_state_dict(torch.load(
            args.load_ckpt, map_location=torch.device('cpu')))
        print(f'Model is loaded weights from {args.load_ckpt}')
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

    human_test_data = MyDataSet(
        name='human_test',
        path=args.human_test_data,
        args=args,
        tokenizer=tokenizer,
        fasttext_model=fasttext_model)

    trainer = Trainer(args=args,
                      model=model,
                      train_data=train_data,
                      dev_data=dev_data,
                      test_data=test_data,
                      human_test_data=human_test_data)
    test_prev = 0
    dev_prev = 0
    htest_prev = 0
    
    # Logging best model
    best_prec = 0
    best_model_dir = 0
    best_iter = 0
    best_epoch = 0
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    print('--------------------- TRAINING ---------------------')

    for i in range(1, args.iteration + 1):
        print(f'Training model on iteration {i} ...')
    
        for e in range(1, args.num_epochs + 1):
            print(f'Training model on epoch {e} of iteration {i}')

            if args.do_train:
                trainer.train()
                print('--------------------- EVALUATING ---------------------')
                htest_prec, htest_recall, htest_f1 = trainer.eval('human_test')
                test_prec, test_recall, test_f1 = trainer.eval('test')
                dev_prec, dev_recal, dev_f1 = trainer.eval('dev')

                if htest_prec > test_prev:
                    # trainer.save_model(f1_score)
                    torch.save(trainer.model.state_dict(),
                               f'./{args.ckpt_dir}/checkpoint_{str(htest_prec)[:5]}.pth')
                    print(f"Save model at {args.ckpt_dir}/checkpoint_{str(htest_prec)[:5]}.pth")
                    test_prev = htest_prec
                    not_update_cnt = 0

                    # Log best model
                    best_iter = i
                    best_epoch = e
                    best_prec = htest_prec
                    best_model_dir = f"{args.ckpt_dir}/checkpoint_{best_prec}.pth"
                else:
                    not_update_cnt += 1
                    if not_update_cnt % 3 == 0:
                        print(f'[INFO] Update learning rate from {trainer.args.learning_rate} => {trainer.args.learning_rate/args.scale_lr}')
                        trainer.update_lr(args.scale_lr) # divide lr to 2
                        print(f'[INFO] Get bestcheckpoint to finetune . Load pretrained model from checkpoint {best_model_dir}')
                        model.load_state_dict(torch.load(
                            best_model_dir, map_location=torch.device('cpu')))
                current_time = str(datetime.datetime.now(pytz.timezone('Asia/Bangkok')))[5:19]
                f = open(args.log_file, "a")
                f.write(','.join([current_time,str(i) , str(e), str(dev_prec)[:5], str(dev_recal)[:5], str(dev_f1)[:5], str(test_prec)[:5], str(test_recall)[:5],str(test_f1), str(htest_prec)[:5], str(htest_recall)[:5], str(htest_f1)[:5]])+'\n')
                # f.write(','.join([current_time,str(i) , str(e), str(dev_prec)[:5], str(dev_recal)[:5],str(test_f1), str(htest_prec)[:5], str(htest_recall)[:5], str(htest_f1)[:5]])+'\n')
                f.close()
                print(f"[INFO] Best test precision : {best_prec} at iter {best_iter}-{best_epoch} is saved at {best_model_dir} ")

def main_aug_online(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if args.use_fasttext:
        fasttext_model = fasttext.load_model(args.fasttext_path)
    else:
        fasttext_model = None
    print('--------------------- MODEL SETTING UP ---------------------')
    if args.use_pretrained:
        print(f'Use pretrained model from checkpoint {args.load_ckpt}')
        model = Model(args)
        model.load_state_dict(torch.load(
            args.load_ckpt, map_location=torch.device('cpu')))
        print(f'Model is loaded weights from {args.load_ckpt}')
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

    human_test_data = MyDataSet(
        name='human_test',
        path=args.human_test_data,
        args=args,
        tokenizer=tokenizer,
        fasttext_model=fasttext_model)

    test_prev = 0
    dev_prev = 0
    not_update_cnt = 0

    # Logging best model
    best_prec = 0
    best_model_dir = 0
    best_iter = 0
    best_epoch = 0
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
            test_data=test_data,
            human_test_data=human_test_data
            )
        for e in range(1, args.num_epochs + 1):
            print(f'Training model on epoch {e} of iteration {i}')
            if args.do_train:
                trainer.train()
                print('--------------------- TESTING ---------------------')
                
                htest_prec, htest_recall, htest_f1 = trainer.eval('human_test')
                test_prec, test_recall, test_f1 = trainer.eval('test', test_prev)
                dev_prec, dev_recal, dev_f1 = trainer.eval('dev', dev_prev)

                if htest_prec > test_prev:
                    # trainer.save_model(f1_score)
                    torch.save(trainer.model.state_dict(),
                               f'./{args.ckpt_dir}/checkpoint_{str(htest_prec)[:5]}.pth')
                    print(f"Save model at {args.ckpt_dir}/checkpoint_{str(htest_prec)[:5]}.pth")
                    test_prev = htest_prec
                    not_update_cnt = 0

                    # Log best model
                    best_iter = i
                    best_epoch = e
                    best_prec = htest_prec
                    best_model_dir = f"{args.ckpt_dir}/checkpoint_{str(best_prec)[:5]}.pth"
                else:
                    not_update_cnt += 1
                    if not_update_cnt % 3 == 0:
                        print(f'[INFO] Update learning rate from {trainer.args.learning_rate} => {trainer.args.learning_rate/args.scale_lr}')
                        trainer.update_lr(args.scale_lr) # divide lr to 2
                        print(f'[INFO] Get bestcheckpoint to finetune . Load pretrained model from checkpoint {best_model_dir}')
                        model.load_state_dict(torch.load(best_model_dir, map_location=torch.device('cpu')))
                current_time = str(datetime.datetime.now(pytz.timezone('Asia/Bangkok')))[5:19]
                f = open(args.log_file, "a")
                f.write(','.join([current_time,str(i) , str(e), str(dev_prec)[:5], str(dev_recal)[:5], str(dev_f1)[:5], str(test_prec)[:5], str(test_recall)[:5],str(test_f1), str(htest_prec)[:5], str(htest_recall)[:5], str(htest_f1)[:5]])+'\n')
                # f.write(','.join([current_time,str(i) , str(e), str(dev_prec)[:5], str(dev_recal)[:5],str(test_f1), str(htest_prec)[:5], str(htest_recall)[:5], str(htest_f1)[:5]])+'\n')
                f.close()
                print(f"[INFO] Best test precision : {best_prec} at iter {best_iter}-{best_epoch} is saved at {best_model_dir} ")

if __name__ == "__main__":
    args = get_config()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using Device : {args.device}')

    if torch.cuda.is_available():
        print(f"GPU device : {torch.cuda.get_device_name(0)}")

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
        
    if args.aug_online:
        print(
            f'[INFO] Train Augmentation Online with epoch = {args.num_epochs} for each iteration = {args.iteration}')
        main_aug_online(args)
    else:
        if args.aug_offline:
            print('[INFO] Train Augmentation Offline ')
        else:
            print('[INFO] Train Normal')
        main(args)
