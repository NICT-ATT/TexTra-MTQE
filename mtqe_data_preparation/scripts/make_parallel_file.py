#!/usr/bin/env python

import sys

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print("Usage: {0} <source corpus> <target corpus> <output corpus>".format(sys.argv[0]))
        exit()
        
    in_source_file, in_target_file, out_paralel_corpus = sys.argv[1:]

    with open(in_source_file, mode = 'r', encoding = 'utf-8') as source_fid, \
        open(in_target_file, mode = 'r', encoding = 'utf-8') as target_fid, \
        open(out_paralel_corpus, mode = 'w', encoding = 'utf-8') as paralel_fid:
            for source_line, target_line in zip(source_fid, target_fid):
                source_line = source_line.strip()
                target_line = target_line.strip()
                if len(source_line) > 0 and len(target_line) > 0:
                    paralel_fid.write("{0} ||| {1}\n".format(source_line, target_line))
