#!/usr/bin/env python

import sys
from xml.sax.saxutils import escape

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Usage: {0}<input corpus> <output corpus>".format(sys.argv[0]))
        exit()
        
    in_corpus, out_corpus = sys.argv[1:]

    with open(in_corpus, mode = 'r', encoding = 'utf-8') as corpus, \
        open(out_corpus, mode = 'w', encoding = 'utf-8') as output:
            for idx, line in enumerate(corpus):
                line = escape(line.strip()).replace('"','\\"')
                output.write("{0} ({1})\n".format(line, idx))
