#!/usr/bin/env python
import argparse
import os
import sys
import numpy as np
import torch
from model import TerminatorTagger, MLM
from dataset import tokenize, collate
from torch.nn import functional as F

import constants
constants.init(1)

import captum
from captum.attr import TokenReferenceBase, LayerIntegratedGradients
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)s] %(message)s')
logger = logging.getLogger("feature attribution")

def main():
    parser = argparse.ArgumentParser(description='evaluate feature importance')
    parser.add_argument('--encoder-config','-ec',default="config/64.8.256.4.json", help="model configuration")
    parser.add_argument('--fasta','-f',type=str,required=True,help="input feature attributions")
    parser.add_argument('--device','-d',default="cuda:0",choices=["cuda:0","cuda:1","cpu"],help="Device to run the model")
    parser.add_argument('--model','-m',default="model/batter.mdl.pt",type=str,help="Where to load the model parameters")
    parser.add_argument('--output','-o',required = True,type=str,help="Where to save feature importance")
    args = parser.parse_args()

    logger.info("Initialize the model ...")
    tagger = TerminatorTagger(MLM(args.encoder_config).encoder)
    logger.info(f"Load model paramters from {args.model} ...")
    state_dict = torch.load(args.model,map_location = args.device)
    tagger.load_state_dict(state_dict)
    tagger.to(args.device)
    tagger.eval()
    sequences = {}
    logger.info("load sequences ...")
    with open(args.fasta) as f:
        for line in f:
            if line.startswith(">"):
                seq_id = line[1:].strip().split(" ")[0]
                sequences[seq_id] = ""
            else:
                sequences[seq_id] += line.strip()

    fout = open(args.output,"w")
    logger.info(f"results will be saved to {args.output} .")
    for seq_id, sequence in sequences.items():
        logger.info(f"processing {seq_id} ...")
        tagger.zero_grad()
        tokens = tokenize(str(sequence).upper().replace("T","U"))
        tokens = collate([tokens])
        tokens = tokens.to(args.device)
        logits = tagger(tokens)[0]
        probabilities = np.round(F.softmax(logits,-1)[0,:,1].cpu().detach().numpy(),3)
        seq_length = tokens.shape[1]
        # (batch_size, seq_length, num_tags)
        token_reference = TokenReferenceBase(reference_token_idx=constants.tokens_to_id[constants.pad_token])
        attention_mask = (tokens != constants.tokens_to_id[constants.pad_token]).int()
        tags, scores = tagger.crf.decode(logits, attention_mask, nbest=1)
        tags = tags[0].squeeze(0).detach().cpu().numpy()
        lig = LayerIntegratedGradients(lambda x: F.softmax(tagger(x)[0],-1).transpose(0,1), tagger.encoder.embeddings)
        reference_tokens = token_reference.generate_reference(seq_length, device=args.device).unsqueeze(0)
        attributions_ig, delta = lig.attribute(tokens, reference_tokens, target = (0,1), n_steps=1000, return_convergence_delta=True,internal_batch_size =128)
        attributions_ig = attributions_ig[0,:,:].sum(dim=-1)
        attributions_ig = attributions_ig / torch.norm(attributions_ig)
        contributions = np.round(attributions_ig.cpu().numpy(),3)[:seq_length]
        print(f">{seq_id}",file=fout)
        print(sequence,file=fout)
        print(",".join(tags[1:-1].astype(str)),file=fout)
        print(",".join(probabilities[1:-1].astype(str)),file=fout)
        print(",".join(contributions[1:-1].astype(str)),file=fout)

    fout.close()
    logger.info("all done .")

if __name__ == "__main__":
    main()
