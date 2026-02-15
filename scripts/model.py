from transformers import LongformerModel, LongformerConfig
import torch
from torch import nn
import numpy as np
import constants
from crf import CRF


class MLM(nn.Module):
    def __init__(self, config):
        super(MLM, self).__init__()
        config = LongformerConfig().from_json_file(config)
        self.encoder = LongformerModel(config)
        self.classifier = nn.Linear(self.encoder.config.hidden_size,len(constants.tokens_list))
    def forward(self, batched_tokens):
        attention_mask = (batched_tokens != constants.tokens_to_id[constants.pad_token]).int()
        global_attention_mask = torch.zeros(batched_tokens.shape, dtype=torch.long, device=batched_tokens.device)
        x = self.encoder(batched_tokens, attention_mask, global_attention_mask).last_hidden_state.relu()
        return self.classifier(x)


class TerminatorTagger(nn.Module):
    def __init__(self,encoder, hidden_dim=64):
        super(TerminatorTagger, self).__init__()
        self.encoder = encoder
        self.linear_1 = nn.Linear(self.encoder.config.hidden_size,hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim,2)
        self.crf = CRF(num_tags=2, batch_first=True)
    def forward(self, batched_tokens, labels=None):
        attention_mask = (batched_tokens != constants.tokens_to_id[constants.pad_token]).long()
        global_attention_mask = torch.zeros(batched_tokens.shape, dtype=torch.int, device=batched_tokens.device)
        x = self.encoder(batched_tokens, attention_mask, global_attention_mask).last_hidden_state.relu()
        x = self.linear_1(x).relu()
        logits = self.linear_2(x)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions = logits, tags=labels, mask=attention_mask)
            outputs =(-1*loss,)+outputs
        return outputs # (loss), scores

    

