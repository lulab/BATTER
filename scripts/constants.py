# define special tokens
pad_token="[PAD]"
unk_token="[UNK]"
cls_token="[CLS]"
sep_token="[SEP]"
mask_token="[MASK]"


# define RNA alphabet
alphbet = "ACGU"

def init(kmer_size):
    from itertools import product
    global kmer_tokens_list
    kmer_tokens_list = []
    # define RNA kmer tokens
    for kmer in product(*[list(alphbet)]*kmer_size):
        kmer_tokens_list.append("".join(kmer))
    global kmer_tokens_set
    kmer_tokens_set = set(kmer_tokens_list)
    global special_tokens_list
    special_tokens_list = [pad_token,unk_token,cls_token,sep_token,mask_token]

    # get token to id mapping
    global tokens_list
    tokens_list = special_tokens_list + kmer_tokens_list
    global tokens_to_id
    tokens_to_id = {}
    for i, token in enumerate(tokens_list):
        tokens_to_id[token] = i

