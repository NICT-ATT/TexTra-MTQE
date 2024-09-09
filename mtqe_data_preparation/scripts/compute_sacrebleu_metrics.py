#!/usr/bin/env python

import argparse
from sacrebleu import BLEU, CHRF

def load_corpora(trans_filename, ref_filename):
    trans_corpus, ref_corpus = [], []
    with open(trans_filename, mode = "r") as trans, open(ref_filename, mode = "r") as ref:
        for t, r in zip(trans, ref):
            t, r = t.strip(), r.strip()
            if t == "" or r == "": t = r = " "
            trans_corpus.append(t)
            ref_corpus.append(r)
    return trans_corpus, ref_corpus

def main(args):
    translation, reference = load_corpora(args.translation, args.reference)
    metric_bleu = BLEU(tokenize = "none", smooth_method = "add-k", effective_order = True)
    metric_chrf = CHRF(word_order = 0, eps_smoothing = False)
    bleu = []
    chrf = []
    for t, r in zip(translation, reference):
        bleu.append("{0:.6f}".format(metric_bleu.sentence_score(t, [r]).score / 100))
        chrf.append("{0:.6f}".format(metric_chrf.sentence_score(t, [r]).score / 100))
    with open("{0}.bleu".format(args.output_prefix), mode = "w") as ob, \
        open("{0}.chrf".format(args.output_prefix), mode = "w") as oc:
            oc.write("\n".join(chrf))
            ob.write("\n".join(bleu))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Tokenize corpora using HuggingFace Transformers tokenizer')

    arg_group = parser.add_argument_group('required arguments')
    arg_group.add_argument('-t', '--translation-corpus', dest = 'translation', help = 'Translation corpus', required = True)
    arg_group.add_argument('-r', '--reference-corpus', dest = 'reference', help = 'Reference corpus', required = True)
    arg_group.add_argument('-o', '--output-prefix', dest = 'output_prefix', help = 'Output prefix', required = True)

    args = parser.parse_args()
    if len(vars(args)) >= 3:
        main(args)
    else:
        parser.print_help()


