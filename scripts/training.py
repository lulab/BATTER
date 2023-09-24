#!/usr/bin/env python
import torch
import constants
constants.init(1)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from model import TerminatorTagger, MLM
import os
import argparse
from torch.optim import AdamW
from dataset import TerminatorSet, collate2

import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)s] %(message)s')
logger = logging.getLogger("train terminator tagger")

from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

import numpy as np

def get_metrics_by_sequence(y_pred,labels,mask):
    # (batch size, seq length)
    # metric by sequence
    min_length = 30
    exist_pred = ((y_pred == 1) & mask).sum(axis=1) > min_length
    exist_real = ((labels == 1) & mask).sum(axis=1) > min_length
    negative_length = mask[~exist_real,:].sum()
    negative_length = negative_length.item()/1000
    TP = (exist_pred & exist_real).sum()
    FP = (exist_pred & (~exist_real)).sum()
    FN = ((~exist_pred) & exist_real).sum()
    TN = ((~exist_pred) & (~exist_real)).sum()
    TP, FP, FN, TN = TP.item(), FP.item(), FN.item(), TN.item()
    precision = TP/(FP+TP) if (FP+TP) > 0 else 0
    recall = TP/(FN+TP) if (FN+TP) > 0 else 0
    FPR = FP/negative_length
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
    return precision, recall, FPR, f1


def get_metrics_by_base(y_pred,labels,mask):
    y_pred = y_pred[mask]
    labels = labels[mask]
    TP = (y_pred == 1) & (labels == 1)
    FP = (y_pred == 1) & (labels == 0)
    FN = (y_pred == 0) & (labels == 1)
    # metric by base
    TP, FP, FN = TP.sum(), FP.sum(), FN.sum()
    TP, FP, FN = TP.item(), FP.item(), FN.item()
    precision = TP/(FP+TP) if (FP+TP) > 0 else 0
    recall = TP/(FN+TP) if (FN+TP) > 0 else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
    return precision, recall, f1



def main():
    parser = argparse.ArgumentParser(description='train terminator tagger')
    parser.add_argument('--encoder-config','-ec',required=True, help="model configuration")
    parser.add_argument('--train-positive','-tp',type=str,required=True,help="positive sequence for training")
    parser.add_argument('--train-group','-tg',type=str,required=True,help="groupping of training instance")
    parser.add_argument('--train-negative','-tn',type=str,help="negative sequence for training")
    parser.add_argument('--strand-supervision', type=float, default=0.0, help="fraction of strand supervision in negative instance sampling")
    parser.add_argument('--val-positive','-vp',type=str,required=True,help="positive sequence for validation")
    parser.add_argument('--val-negative','-vn',type=str,help="negative sequence for validation")
    parser.add_argument('--val-group','-vg',type=str,required=True,help="groupping of validation instance")
    parser.add_argument('--negative-fraction','-nf',type=float,default=0.2,help="negative fraction in training")
    parser.add_argument('--shuffled-fraction','-sf',type=float,default=0.5,help="shuffling this fraction in negative instances")
    parser.add_argument('--shuffled-order','-so',type=int,default=4,help="kmer to preserve in shuffling")
    parser.add_argument('--batch-size', '-bs', type=int,default=64,help="batch size for scanning")
    parser.add_argument('--weight-decay', '-wd', type=float, default=0.01,help="weight decay to use")
    parser.add_argument('--warmup-steps',default=0,type=int,help="number of steps for learning rate warm up")
    parser.add_argument('--device', '-d', default="cuda:0", choices=["cuda:0","cuda:1","cpu"],help="Device to run the model")
    parser.add_argument('--models', '-m', required=True, type=str, help="directory to save model check points")
    parser.add_argument('--pretrained-model', '-pm', help="whether start from pretrained model .")
    parser.add_argument('--resume-from', '-rf', help="resume training from an existed model .")
    parser.add_argument('--max-grad-norm',default=1,type=float,help="gradient clipping")
    parser.add_argument('--metrics', help="where to save training metrics .")
    parser.add_argument('--epoches', '-e', type=int, default=256, help="Number of epoches to train")
    parser.add_argument('--min-length', '-ml', type=int, default=64, help="Minimal length of sampled sequence for training")
    parser.add_argument('--learning-rate', '-lr', type=float, default=5e-5, help="Learning rate ro use")
    args = parser.parse_args()
    device = args.device

    logger.info(f"Model checkpoints will be saved to {args.models} .")


    if not os.path.exists(args.models):
        logger.warn(f"the directory {args.models} does not exists, create it .")
        os.makedirs(args.models)
    pretranied_model = MLM(args.encoder_config)    
    if args.resume_from is not None:
        logger.info(f"Resume model training from {args.resume_from} ...")
        tagger = TerminatorTagger(pretranied_model.encoder)
        state_dict = torch.load(args.resume_from, map_location = args.device)
        tagger.load_state_dict(state_dict)
    else:
        if args.pretrained_model is not None:
            logger.info("Intialize terminator tagger with pretrained encoder ...")
            state_dict = torch.load(args.pretrained_model,map_location = args.device) 
            pretranied_model.load_state_dict(state_dict)
        tagger = TerminatorTagger(pretranied_model.encoder)
    tagger.to(device)


    logger.info("Load training set ...")
    train_set = TerminatorSet(args.train_positive,args.train_group,args.train_negative,
                              negative_fraction = args.negative_fraction, shuffled_fraction = args.shuffled_fraction,
                              shuffled_order = args.shuffled_order, antisense_supervision = args.strand_supervision, min_length = args.min_length)
    train_sampler = RandomSampler(train_set)
    train_loader = DataLoader(train_set, sampler=train_sampler,batch_size=args.batch_size, collate_fn=collate2)


    logger.info("Load validation set ...")
    val_set = TerminatorSet(args.val_positive,args.val_group,args.val_negative,
                            negative_fraction = args.negative_fraction, shuffled_fraction = args.shuffled_fraction, antisense_supervision=0)
    val_sampler = RandomSampler(val_set)
    val_loader = DataLoader(val_set, sampler=val_sampler, batch_size=args.batch_size, collate_fn=collate2)
    
    # not apply weight decay to bias and layerNorm weight
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [{
        "params": [p for n, p in tagger.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
        },{
        "params": [p for n, p in tagger.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0},]
    
    optimizer = AdamW(optimizer_grouped_parameters,lr=args.learning_rate, eps=1e-8, betas=(0.9,0.999))
    total_steps = len(train_loader)*args.epoches
    #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)

    logger.info("start training ...")    

    fp = open(args.metrics,"w")
    #train_losses, train_precisions, train_recalls, train_f1s = [], [], [], []
    train_losses = []
    i = 0
    for e in range(args.epoches):
        for batched_tokens, bacthed_labels in train_loader:
            batched_tokens, bacthed_labels = batched_tokens.to(device), bacthed_labels.to(device)
            optimizer.zero_grad()
            loss, logits = tagger(batched_tokens, bacthed_labels)
            train_losses.append(loss.item())
            """
            mask = batched_tokens != constants.tokens_to_id[constants.pad_token] 
            y_pred_crf, _  = tagger.crf.decode(logits, mask)
            precision, recall, f1 = get_metrics_by_sequence(y_pred_crf[0], bacthed_labels, mask)
            train_precisions.append(precision)
            train_recalls.append(recall)
            train_f1s.append(f1)
            """
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tagger.parameters(), args.max_grad_norm) # apply gradient clipping
            optimizer.step()
            scheduler.step() # update learning rate
            i += 1
            if i % 512 == 0:
                j = 0
                tagger.eval()
                #print(e,i,"train",np.mean(train_losses),np.mean(train_precisions),np.mean(train_recalls),np.mean(train_f1s),sep="\t",file=fp)
                print(e,i,"train",np.mean(train_losses),file=fp,sep="\t")
                #train_losses, train_precisions, train_recalls, train_f1s = [], [], [], []
                train_losses = []
                val_precisions, val_recalls, val_fprs, val_f1s = [], [], [], []
                for batched_tokens, bacthed_labels in val_loader:
                    batched_tokens, bacthed_labels = batched_tokens.to(device), bacthed_labels.to(device)
                    j += 1
                    logits = tagger(batched_tokens)[0]
                    mask = batched_tokens != constants.tokens_to_id[constants.pad_token]
                    y_pred_crf, _ = tagger.crf.decode(logits, mask)
                    precision, recall, fpr, f1 = get_metrics_by_sequence(y_pred_crf[0], bacthed_labels, mask)
                    val_precisions.append(precision)
                    val_recalls.append(recall)
                    val_fprs.append(fpr)
                    val_f1s.append(f1)
                    if j == 16:
                        break
                print(e,i,"validation",np.mean(val_precisions),np.mean(val_recalls),np.mean(val_fprs),np.mean(val_f1s),sep="\t",file=fp)
                fp.flush()
                tagger.train()
        torch.save(tagger.state_dict(),f"{args.models}/{e}.pt")
    fp.close()
if __name__ == "__main__":
    main()
