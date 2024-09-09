#!/usr/bin/env python

import sys
import argparse
import numpy as np
from scipy.stats import pearsonr, spearmanr

def load_file(filename):
    scores = []
    with open(filename, mode = "r") as f:
        for line in f:
            line = line.strip().split()
            if len(line) > 1:
                line = line[2]
            else:
                line = line[0]
            scores.append(np.float(line))
    return np.asarray(scores)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('system', help='System file')
    parser.add_argument('gold', help='Gold output file')

    args = parser.parse_args()

    ref = load_file(args.gold)
    pred = load_file(args.system)

    assert len(ref) == len(pred), \
        "Incorrect system predictions, expect {0}, got {1}.".format(len(ref), len(pred))

    pearson = pearsonr(ref, pred)[0]
    spearman = spearmanr(ref, pred)[0]
    diff = ref - pred
    mae = np.abs(diff).mean()
    rmse = (diff ** 2).mean() ** 0.5

    print("pearson: {0:.4f}".format(pearson))
    print("spearman: {0:.4f}".format(spearman))
    print("mae: {0:.4f}".format(mae))
    print("rmse: {0:.4f}".format(rmse))

