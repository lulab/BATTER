#!/usr/bin/env python
import torch
import constants
constants.init(1)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW
from model import MLM
from dataset import RNASet, collate, get_masked_tokens

import argparse
import os
import sys
import numpy as np

from transformers import get_linear_schedule_with_warmup #, AdamW


import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)s] %(message)s')
logger = logging.getLogger("masked language pretraining")



def main():
    parser = argparse.ArgumentParser(description='perform masked language pretraining')
    parser.add_argument('--model-config','-mc',required=True, help="model configuration")
    parser.add_argument('--train-sequence','-ts',required=True,help="training sequences")
    parser.add_argument('--train-group','-tg',required=True,help="training grouping")
    parser.add_argument('--val-sequence','-vs',required=True,help="validation sequences")
    parser.add_argument('--val-group','-vg',required=True,help="validation grouping")
    parser.add_argument('--models','-m',required=True,help="Directory to saving models")
    parser.add_argument('--performance','-p',required=True,help="Where to save model performance")
    parser.add_argument('--weight-decay','-wd',default=0.01,type=float,help="weight decay")
    parser.add_argument('--device','-d',default="cuda:0",choices=["cuda:0","cuda:1","cpu"],help="Device to run the model")
    parser.add_argument('--epoches','-e',default=2048,type=int,help="Number of epoches for training")
    parser.add_argument('--warmup-steps',default=0,type=int,help="number of steps for learning rate warm up")
    parser.add_argument('--batch-size','-bs',default=64,type=int,help="batch size for training")
    parser.add_argument('--max-grad-norm',default=1,type=float,help="gradient clipping")
    args = parser.parse_args()
    device = args.device
    logger.info("initialize model ...")
    model = MLM(args.model_config)
    model = model.to(device)

    if not os.path.exists(args.models):
        logger.info(f"{args.models} not exists, create one .")
        os.mkdir(args.models)

    logger.info("load training data ...")

    train_set = RNASet(args.train_sequence,args.train_group)
    train_sampler = RandomSampler(train_set)
    train_loader = DataLoader(train_set, sampler=train_sampler,batch_size=args.batch_size, collate_fn=collate)

    logger.info("load validation data ...")
    val_set = RNASet(args.val_sequence,args.val_group)
    val_sampler = RandomSampler(val_set)
    val_loader = DataLoader(val_set, sampler=val_sampler,batch_size=args.batch_size, collate_fn=collate)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [{
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": 0,
        },{
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay},]
    
    optimizer = AdamW(optimizer_grouped_parameters,lr=5e-5, eps=1e-8, betas=(0.9,0.999))
    total_steps = int(len(train_loader)*args.epoches)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    logger.info(f"training metrics will be saved to {args.performance} .")
    flog = open(args.performance,"w")
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    for e in range(args.epoches):
        i = 0
        train_accuracies = []
        for batched_tokens in train_loader:
            optimizer.zero_grad()
            batched_tokens = batched_tokens.to(device)
            batched_tokens, labels = get_masked_tokens(batched_tokens)
            logits = model(batched_tokens)
            loss = criterion(logits.transpose(2,1),labels)
            accurate = logits.argmax(axis=-1) == labels
            accurate = accurate[labels!=-100].reshape(-1)
            accuracy = accurate.sum()/accurate.shape[0]
            train_accuracies.append(accuracy.item())
            if i%512 == 0:
                val_accuracies = []
                model.eval()
                j = 0
                for batched_tokens in val_loader:
                    if j >= 32:
                        break
                    batched_tokens = batched_tokens.to(device)
                    batched_tokens, labels = get_masked_tokens(batched_tokens)
                    logits = model(batched_tokens)
                    accurate = logits.argmax(axis=-1) == labels
                    accurate = accurate[labels!=-100].reshape(-1)
                    accuracy = accurate.sum()/accurate.shape[0]
                    val_accuracies.append(accuracy.item())
                    j += 1
                print(e,i,np.mean(train_accuracies),np.mean(val_accuracies),sep="\t",file=flog)
                flog.flush()
                train_accuracies = []
                model.train()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            i += 1
        if e%10 == 0:
            logger.info(f"{e} epoches passed, save a model check point .")
            torch.save(model.state_dict(),f"{args.models}/{e}.{i}.pt")
    flog.close()
if __name__ == "__main__":
    main()
