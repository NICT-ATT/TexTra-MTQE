#!/usr/bin/env python
# -*- coding: utf-8 -*-

# General imports
import sys
import argparse
import logging
import gc
import numpy as np

# Import pytorch 
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import huggingface's packages and set logging level
import transformers
logging.getLogger("transformers").setLevel(logging.ERROR)
from transformers import XLMRobertaTokenizerFast
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

# Import MTQE
from nn.MTQE import MTQE

# The tokenize function takes as input a tokenizer, source and target sequences, and returns the prepared model input
def tokenize(tokenizer, src, trg):
    model_inputs = tokenizer(
        text = [src],
        text_pair = [tgt],
        return_attention_mask = True,
        return_special_tokens_mask = True,
        truncation = False,
        max_length = None,
        return_tensors = "pt",
    )
    return model_inputs

def main(model, tokenizer, device, src, tgt):
    # Remove spaces from beginning and end of source and target sequences
    src, tgt = src.strip(), tgt.strip()
    # Tokenize the source and target sequences
    batch = tokenize(tokenizer, src, tgt)
    # Move the tokenizer output to GPU
    for item in batch: batch[item] = batch[item].to(device)
    # Do a forward pass to the model to get the MTQE outputs
    with torch.no_grad():
        outputs = model(batch["input_ids"], batch["attention_mask"], batch["special_tokens_mask"])
    # Retrieve the input sequence IDs as processed by the tokenizer
    seq = batch["input_ids"][0].tolist()
    # Apply the softmax function to the token-level QE predictions to normalize scores between 0 and 1
    token_soft = torch.softmax(outputs["TOKEN"], dim = 1)[0][1]
    # Initialize two empty lists, one for source and one for target sequences, in which we will append tokens and scores
    src_token = []
    tgt_token = []
    # Initialize a pointer to know if we are in the source or in the target sequence
    in_tgt = 0
    # Iterate over the input sequence (source and target concatenated) tokens
    for idx, token_id in enumerate(seq):
        # Map the ID to the token string based on the tokenizer
        token = tokenizer.convert_ids_to_tokens(token_id)
        # If we encounter the end of sequence token, it means the source sequence is over
        if token == "</s>" and in_tgt == 0: in_tgt = 1
        # If the current token is not a BOS, EOS and BPE token
        if token not in noprint:
            # We replace the BPE symbol by empty string for printing purposes
            token = token.replace('▁', '')
            # If we are in the source sequence, add the token and the token-level QE BAD class score
            if in_tgt == 0:
                src_token.append("({0} | {1:.5f})".format(token, token_soft[idx]))
            # Else, we do the same for the target sequence
            else:
                tgt_token.append("({0} | {1:.5f})".format(token, token_soft[idx]))

    # Prepare each QE result for output
    ter = "{0:.5f}".format(outputs["TER"].item())
    bleu = "{0:.5f}".format(outputs["BLEU"].item())
    chrf = "{0:.5f}".format(outputs["CHRF"].item())
    src_token = " ".join(src_token)
    tgt_token = " ".join(tgt_token)
    # Print final QE results
    print("TER: {0}\tchrF: {1}\tBLEU: {2}\tSource: {3}\tTarget: {4}".format(ter, chrf, bleu, src_token, tgt_token))

if __name__ == "__main__":

    # Define our argument parser, we only require a pretrained MTQE model
    parser = argparse.ArgumentParser(description = 'Inference script for MTQE')
    arg_group = parser.add_argument_group('required arguments')
    arg_group.add_argument('-m', '--model', dest = 'model', help = 'Pretrained MTQE model', type = str, required = True)
    args = parser.parse_args()

    # Define the tokens not to be printed
    noprint = ["<s>", "</s>", "▁"]
    # Define the model name, MTQE is based on XLM-Roberta (could be large or other, depending on what was used during training)
    model_name = 'xlm-roberta-large'
    # Define the device to be used, here we use GPU
    device = torch.device("cuda")
    # Instantiate the Huggingface Tokenizer given a model name
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_name)
    # Instantiate the MTQE model given a model name
    model = MTQE(model_name)
    # Load the model parameters based on the pretrained model given as argument
    model.load_state_dict(torch.load(args.model))
    # Switch the model to evaluation mode
    model.eval()
    # Move the model to GPU
    model = model.to(device)

    # Loop over lines given to stdin
    for line in sys.stdin:
        # Split line in source and target sequences, as they as separated by tab
        elements = line.strip().split("\t")
        src = elements[0]
        tgt = elements[1] if (len(elements) >= 2) else ""
        # Call the main function with the arguments (the path to the QE model), the source and the target sequences
        main(model, tokenizer, device, src, tgt)
