#!/usr/bin/env python
# coding=utf-8

import sys
import argparse
import json

def main(args):
    with open(args.source, mode = "r") as src, open(args.translation, mode = "r") as tgt, \
        open(args.s_tok, mode = "r") as src_tok, open(args.t_tok, mode = "r") as tgt_tok, \
        open(args.s_tags, mode = "r") as src_tags, open(args.t_tags, mode = "r") as tgt_tags, \
        open(args.bleu, mode = "r") as b, open(args.ter, mode = "r") as t, open(args.chrf, mode = "r") as c, \
        open(args.output, mode = "w") as o:
        sent_index = 0
        for src_line, tgt_line, src_tok_line, tgt_tok_line, src_tags_line, tgt_tags_line, b_line, t_line, c_line \
            in zip(src, tgt, src_tok, tgt_tok, src_tags, tgt_tags, b, t, c):
            sent_index += 1
            if len(src_tok_line.split()) + len(tgt_tok_line.split()) < args.max_len \
                and len(src_tok_line.split()) == len(src_tags_line.split()) \
                and len(tgt_tok_line.split()) == len(tgt_tags_line.split()):
                if float(t_line) > 1.0: t_line = 1.0
                o.write("{0}\n".format(\
                    json.dumps(\
                        {"source": src_line.strip(), "target": tgt_line.strip(), \
                        "source_tags": src_tags_line.strip(), "target_tags": tgt_tags_line.strip(), \
                        "bleu": float(b_line), "ter": float(t_line), "chrf": float(c_line) \
                        }) \
                    ) \
                )
            else:
                print("Too lengthy sentence pair in line {0} is skipped".format(sent_index))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Produce JSON file from all MTQE related data')

    arg_group = parser.add_argument_group('required arguments')
    arg_group.add_argument('-s', '--source-corpus', dest = 'source', help = 'Source corpus', required = True)
    arg_group.add_argument('-t', '--translation-corpus', dest = 'translation', help = 'Translation corpus', required = True)
    arg_group.add_argument('-stok', '--source-tokenized', dest = 's_tok', help = 'Source corpus tokenized', required = True)
    arg_group.add_argument('-ttok', '--translation-tokenized', dest = 't_tok', help = 'Translation corpus tokenized', required = True)
    arg_group.add_argument('-stag', '--source-tags', dest = 's_tags', help = 'Source tags', required = True)
    arg_group.add_argument('-ttag', '--target-tags', dest = 't_tags', help = 'Target tags', required = True)
    arg_group.add_argument('-bleu', '--bleu', dest = 'bleu', help = 'BLEU', required = True)
    arg_group.add_argument('-ter', '--ter', dest = 'ter', help = 'TER', required = True)
    arg_group.add_argument('-chrf', '--chrf', dest = 'chrf', help = 'ChrF', required = True)
    arg_group.add_argument('-ml', '--max-length', dest = 'max_len', help = 'Maximum source and target concatenated sequences length', \
                            type = int, required = True)
    arg_group.add_argument('-o', '--output-file', dest = 'output', help = 'Output JSON file created', required = True)

    args = parser.parse_args()
    if len(vars(args)) >= 11:
        main(args)
    else:
        parser.print_help()
