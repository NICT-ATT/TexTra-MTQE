#!/usr/bin/env python

import argparse
import logging
from transformers import XLMRobertaTokenizerFast
from datasets import Dataset
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

def process_corpus(corpus, tokenizer):
    def tokenize_dataset(samples):
        content = tokenizer(samples["line"], add_special_tokens = False, padding = False, truncation = False, \
            max_length = None, stride = 0, is_split_into_words = False, pad_to_multiple_of = None, return_tensors = None, \
            return_token_type_ids = False, return_attention_mask = False, return_overflowing_tokens = False, \
            return_special_tokens_mask = False, return_offsets_mapping = False, return_length = False, verbose = False, \
        )
        tokenized = []
        for line in content["input_ids"]:
            tmp_line = ""
            for token in line:
                tmp_line += "{0} ".format(tokenizer.convert_ids_to_tokens(token))
            tokenized.append(tmp_line.strip())
        return {"tokenized": tokenized}
    tokenized_content = corpus.map(tokenize_dataset, batched = True, remove_columns = ["line"])
    return tokenized_content

def load_corpus(filename):
    with open(filename, mode = "r") as file_handler:
        for line in file_handler:
            yield {"line": line.strip()}

def write_files(source, translation, reference, output_prefix):
    with open("{0}.source.tok".format(output_prefix), mode = "w") as o_source, \
        open("{0}.translation.tok".format(output_prefix), mode = "w") as o_translation, \
        open("{0}.reference.tok".format(output_prefix), mode = "w") as o_reference:
        for idx, (s, t, r) in enumerate(zip(source["tokenized"], translation["tokenized"], reference["tokenized"])):
            if len(s) > 0 and len(t) > 0 and len(r) > 0:
                o_source.write("{0}\n".format(s))
                o_translation.write("{0}\n".format(t))
                o_reference.write("{0}\n".format(r))

def run_tokenization(corpus_filename, tokenizer, output_prefix):
    corpus = Dataset.from_generator(load_corpus, gen_kwargs = {"filename": corpus_filename})
    corpus = process_corpus(corpus, tokenizer)
    return corpus

def main(args):
    tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-large')
    source = run_tokenization(args.source, tokenizer, args.output_prefix)
    translation = run_tokenization(args.translation, tokenizer, args.output_prefix)
    reference = run_tokenization(args.reference, tokenizer, args.output_prefix)
    write_files(source, translation, reference, args.output_prefix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Tokenize corpora using HuggingFace Transformers tokenizer')

    arg_group = parser.add_argument_group('required arguments')
    arg_group.add_argument('-s', '--source-corpus', dest = 'source', help = 'Source corpus', required = True)
    arg_group.add_argument('-t', '--translation-corpus', dest = 'translation', help = 'Translation corpus', required = True)
    arg_group.add_argument('-r', '--reference-corpus', dest = 'reference', help = 'Reference corpus', required = True)
    arg_group.add_argument('-o', '--output-prefix', dest = 'output_prefix', help = 'Output prefix', required = True)

    args = parser.parse_args()
    if len(vars(args)) >= 4:
        main(args)
    else:
        parser.print_help()

