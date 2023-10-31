import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import logging
from tqdm import tqdm
import sys
import constants
import re
import sys
import random
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(name)s] %(message)s')
logger = logging.getLogger("load sequence instances")

#constants.init(1)

def tokenize(sequence,k=1):
    """
    input: RNA sequence of length L
    output: integer list of shape  L-k+1 + 2
            first element is cls token
            last token is seq token
    """
    tokens = [constants.tokens_to_id[constants.cls_token]]
    for i in range(len(sequence)-k+1):
        kmer = sequence[i:i+k]
        if kmer in constants.kmer_tokens_set:
            tokens.append(constants.tokens_to_id[kmer])
        else:
            tokens.append(constants.tokens_to_id[constants.unk_token])
    tokens.append(constants.tokens_to_id[constants.sep_token])
    return torch.tensor(tokens)

def collate(examples):
    return pad_sequence(examples, batch_first=True, padding_value=constants.tokens_to_id[constants.pad_token])

def collate2(examples):
    batched_tokens = []
    batched_tags = []
    for tokens, tags in examples:
        batched_tokens.append(tokens)
        batched_tags.append(tags)
    batched_tokens = pad_sequence(batched_tokens, batch_first=True, padding_value=constants.tokens_to_id[constants.pad_token])
    batched_tags = pad_sequence(batched_tags, batch_first=True, padding_value=0)
    #for token in constants.special_tokens_list:
    #    batched_tags[batched_tokens==token] = 0
    batched_tags = batched_tags.long()
    return batched_tokens, batched_tags

def get_masked_tokens(batched_tokens, mlm_probability=0.2):
    device = batched_tokens.device
    probability_matrix = torch.full(batched_tokens.shape, mlm_probability)
    special_token_mask = torch.full(batched_tokens.shape,False)
    for token in constants.special_tokens_list:
        special_token_mask[batched_tokens==constants.tokens_to_id[token]] = True
    probability_matrix[special_token_mask] = 0
    mlm_mask = torch.bernoulli(probability_matrix).bool()
    labels = batched_tokens.detach().clone()
    labels[~mlm_mask] = -100
    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    masking_mask = torch.bernoulli(torch.full(batched_tokens.shape, 0.8)).bool() & mlm_mask
    batched_tokens[masking_mask] = constants.tokens_to_id[constants.mask_token]
    # 10% of the time, we replace masked input tokens with random word
    random_mask = torch.bernoulli(torch.full(batched_tokens.shape, 0.5)).bool() & mlm_mask & (~masking_mask)
    random_words = torch.randint(len(constants.special_tokens_list),len(constants.tokens_list), batched_tokens.shape, dtype=torch.long,device=device)
    batched_tokens[random_mask] = random_words[random_mask]
    # 10% of the time, left as is
    return batched_tokens, labels

def load_fasta(path):
    """
    Load fasta file into an sequence dict
    Each sequence records could span multiple lines
    """
    sequences = {}
    attrs = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if len(line)==0:
                continue
            if line.startswith(">"):
                seq_id0= line.replace(">","").strip()
                if " " in seq_id0:
                    p = seq_id0.find(" ")
                    seq_id = seq_id0[:p]
                    attr = seq_id0[p+1:]
                    attrs[seq_id] = attr
                else:
                    seq_id = seq_id0
                sequences[seq_id] = ""
            else:
                sequences[seq_id] += line.upper().replace("T","U")
    return sequences, attrs

def load_group(path):
    grouped_seq_ids = {}
    with open(path) as f:
        for line in f:
            seq_id, group_id = line.strip().split("\t")
            if group_id not in grouped_seq_ids:
                grouped_seq_ids[group_id] = []
            grouped_seq_ids[group_id].append(seq_id)
    return grouped_seq_ids

class RNASet(Dataset):
    def __init__(self, fasta, group, max_length=510):
        self.max_length =  max_length
        logger.info("load sequence ...")
        sequences, _  = load_fasta(fasta) 
        logger.info("load groupping ...")
        grouped_seq_ids = load_group(group)
        cluster_ids = sorted(list(grouped_seq_ids.keys()))
        self.sequences = []
        n_sequence = 0
        logger.info("summarize instances ...")
        for cluster_id in cluster_ids:
            sequences_by_cluster = []
            for seq_id in grouped_seq_ids[cluster_id]:
                if seq_id not in sequences:
                    continue
                sequence = sequences[seq_id]
                sequences_by_cluster.append(sequence)
            if len(sequences_by_cluster) == 0:
                continue
            n_sequence += len(sequences_by_cluster)
            self.sequences.append(sequences_by_cluster)
        n_clusters = len(self.sequences)
        logger.info(f"{n_sequence} sequences are loaded into {n_clusters} clusters .")
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self,idx):
        cluster_size = len(self.sequences[idx])
        cidx = np.random.randint(cluster_size)
        sequence = self.sequences[idx][cidx]
        if len(sequence) > self.max_length:
            start = np.random.randint(len(sequence)-self.max_length)
            sequence = sequence[start:start+self.max_length]
        tokens = tokenize(sequence)
        return tokens

class TerminatorSet(Dataset):
    def __init__(self, terminator, group, background=None, 
                 min_length = 64, max_length=510, antisense_supervision = 0.02,
                 negative_fraction=0.2, shuffled_fraction=0.5, shuffled_order=4):
        """
        terminator: positive sequence contains terminator 
        group: a tabale specifies how to stratify the sequence
        background: negative sequence are randomly sampled from bacteria genome, or created by shuffling positive sequence
        negative_fraction: possibility of sample a negative sequence
        shuffled_fraction: for negative instance, how many instances are generated by shuffling real world sequence instead of using random real world sequence
                           if background if not provided, this parameter is ignored, all negative sequence will be generated by shuffling positive instance
        shuffled_order: kmer counts to preserve, 4 by default
        """
        from ushuffle import shuffle
        from Bio.Seq import Seq
        self.max_length =  max_length
        self.min_length = min_length
        self.negative_fraction = negative_fraction
        self.shuffled_fraction = shuffled_fraction
        self.antisense_supervision = antisense_supervision
        self.shuffled_order = shuffled_order
        logger.info("Load positive sequence ...")
        sequences, attrs  = load_fasta(terminator) 
        # ains: antisense is negative
        locs, ains = {}, {}
        for seq_id in attrs:
            attr  = attrs[seq_id].strip()
            if (" " not in attr) or (antisense_supervision == 0):
                locs[seq_id] = attr.split(" ")[0]
                ains[seq_id] = 0
            else:
                loc, ain = attr.split(" ")
                locs[seq_id] = loc
                ains[seq_id] = ain
        if background is not None:
            logger.info("background sequence provided")
            logger.info("Load negative sequence ...")
            backgrounds, _  = load_fasta(background)
        else:
            logger.info("background sequence not provided, will shuffle positive sequence as background")
        # sequence, terminator locations and otu ids grouped by terminator
        self.sequences = [] # list of list of str
        self.spans = []  # list of list of list of (int,int)
        self.ains = []
        self.otu_ids = [] # list of list of str

        logger.info("load groupping ...")
        grouped_seq_ids = load_group(group)
        cluster_ids = sorted(list(grouped_seq_ids.keys()))

        n_sequence = 0
        n_cluster = 0
        logger.info("Processing positive data ...")
        for cluster_id in cluster_ids:
            sequences_by_cluster = []
            spans_by_cluster = []
            ains_by_cluster = []
            otu_ids_by_cluster = []
            for seq_id in grouped_seq_ids[cluster_id]:
                otu_id = seq_id.split(":")[0]
                if seq_id not in sequences:
                    continue
                sequence = sequences[seq_id]
                if len(sequence) < self.min_length:
                    continue
                sequences_by_cluster.append(sequence)
                spans = []
                for span in locs[seq_id].split(";"):
                    #try:
                    s, e = span.split(",")
                    #except:
                    #print(seq_id, locs[seq_id])
                    #sys.exit(1)
                    s, e = int(s), int(e)
                    spans.append((s, e))
                spans_by_cluster.append(spans)
                ains_by_cluster.append(int(ains[seq_id]))
                otu_ids_by_cluster.append(otu_id)
            assert len(sequences_by_cluster) == len(spans_by_cluster) == len(otu_ids_by_cluster) == len(ains_by_cluster)
            if len(sequences_by_cluster) == 0:
                continue
            n_sequence += len(sequences_by_cluster)
            n_cluster += 1
            self.sequences.append(sequences_by_cluster)
            self.spans.append(spans_by_cluster)
            self.ains.append(ains_by_cluster)
            self.otu_ids.append(otu_ids_by_cluster)
        logger.info(f"{n_sequence} sequences are loaded into {n_cluster} clusters .")
        self.backgrounds = defaultdict(list)
        if background is not None:
            logger.info("Processing negative data ...")
            for seq_id in backgrounds:
                sequence = backgrounds[seq_id]
                if len(sequence) < self.min_length:
                    continue
                otu_id = seq_id.split(":")[0]
                self.backgrounds[otu_id].append(sequence)
        self.bg_otu_ids = list(self.backgrounds.keys())

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self,idx):
        cluster_size = len(self.sequences[idx])
        cidx = np.random.randint(cluster_size)
        if np.random.rand() >= self.negative_fraction: 
            # take a positive instance
            sequence = self.sequences[idx][cidx]
            spans = self.spans[idx][cidx]
        else: # take a negative instance
            if len(self.bg_otu_ids) == 0:
                sequence = self.sequences[idx][cidx]
                sequence = shuffle(sequence.encode(),self.shuffled_order).decode()
            elif (np.random.rand() < self.antisense_supervision) and (self.ains[idx][cidx] == 1):
                # for sequence where antisense is negative
                sequence = self.sequences[idx][cidx]
                sequence = str(Seq(sequence).reverse_complement())
            else:
                otu_id = self.otu_ids[idx][cidx]
                if otu_id not in self.backgrounds:
                    otu_id = np.random.choice(self.bg_otu_ids) 
                oidx = np.random.randint(len(self.backgrounds[otu_id]))
                sequence = self.backgrounds[otu_id][oidx]
                if np.random.rand() < self.shuffled_fraction:
                    # shuffle random sequence with probability 
                    sequence = shuffle(sequence.encode(),self.shuffled_order).decode()
            spans = []
        tags = torch.zeros(len(sequence)) 
        for s, e in spans:
            tags[s:e] = 1
        length = np.random.randint(self.min_length, self.max_length)
        if length >= len(sequence):
            length = len(sequence)
            offset = 0
        else:
            offset = np.random.randint(len(sequence) - length)
        tags = tags[offset:offset+length]
        sequence = sequence[offset:offset+length]
        tokens = tokenize(sequence) # add cls and sep token
        tags = torch.cat((torch.tensor([0]),tags,torch.tensor([0]))) 
        return tokens, tags

