#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

"""
Script to evaluate outputs of word-level machine translation quality estimation systems 
in the WMT 2019 format.
"""

def read_tags(filename):
    all_tags = []
    with open(filename, mode = "r") as f:
        for line in f:
            all_tags.append([item.strip() for item in line.split() if len(item.strip()) > 0])
    return all_tags

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('system', help='System file')
    parser.add_argument('gold', help='Gold output file')
    args = parser.parse_args()

    system_tags = read_tags(args.system)
    gold_tags = read_tags(args.gold)

    assert len(system_tags) == len(gold_tags), 'Number of lines in system and gold file differ'

    # true/false positives/negatives
    tp = tn = fp = fn = 0
    # id to classes
    labels = {1: "BAD", 0: "OK"}

    for i, (sys_sentence, gold_sentence) in enumerate(zip(system_tags, gold_tags)):

        if len(sys_sentence) != len(gold_sentence):
            print('Number of tags in system and gold file differ in line %d' % i)
        else:
            for sys_tag, gold_tag in zip(sys_sentence, gold_sentence):
                if sys_tag == 'OK':
                    if sys_tag == gold_tag:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if sys_tag == gold_tag:
                        tn += 1
                    else:
                        fn += 1

    total_tags = tp + tn + fp + fn
    num_sys_ok = tp + fp
    num_gold_ok = tp + fn
    num_sys_bad = tn + fn
    num_gold_bad = tn + fp

    precision_ok = tp / num_sys_ok if num_sys_ok else 1.
    recall_ok = tp / num_gold_ok if num_gold_ok else 0.
    precision_bad = tn / num_sys_bad if num_sys_bad else 1.
    recall_bad = tn / num_gold_bad if num_gold_bad else 0.

    if precision_ok + recall_ok:
        f1_ok = 2 * precision_ok * recall_ok / (precision_ok + recall_ok)
    else:
        f1_ok = 0.

    if precision_bad + recall_bad:
        f1_bad = 2 * precision_bad * recall_bad / (precision_bad + recall_bad)
    else:
        f1_bad = 0.

    f1_mult = f1_ok * f1_bad

    print('P OK: %.4f' % precision_ok)
    print('R OK: %.4f' % recall_ok)
    print('F1 OK: %.4f' % f1_ok)
    print('P BAD: %.4f' % precision_bad)
    print('R BAD: %.4f' % recall_bad)
    print('F1 BAD: %.4f' % f1_bad)

    print('*****')
    print('F1 Mult: %.4f' % f1_mult)

    mcc_numerator = (tp * tn) - (fp * fn)
    mcc_denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    if mcc_numerator > 0.0 and mcc_denominator > 0.0:
        mcc = mcc_numerator / mcc_denominator
    else:
        mcc = 0.0
    print('Matthews correlation: %.4f' % mcc)

