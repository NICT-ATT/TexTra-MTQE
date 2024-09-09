#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script is a modified version of the code
from https://github.com/clab/fast_align
"""

import os
import sys
import argparse
import subprocess

class Aligner:

    def __init__(self, fast_align_build, corpus, fwd_params, fwd_err, rev_params, rev_err, heuristic, out_filename):
        self.fast_align_build = fast_align_build
        self.corpus = corpus
        self.out_filename = out_filename
        self.fwd_params = fwd_params
        self.fwd_err = fwd_err
        self.rev_params = rev_params
        self.rev_err = rev_err
        self.heuristic = heuristic
        self.set_paths()
        self.set_filenames()
        self.read_err_files()

    def write_result(self):
        if self.alignment is not None:
            with open(self.out, mode = "w") as f:
                f.write(self.alignment)
        else:
            print("Error: final alignment failed")
            exit()

    def set_filenames(self):
        self.fwd_out = "{0}.fwd".format(self.out_filename)
        self.rev_out = "{0}.rev".format(self.out_filename)
        self.out = "{0}".format(self.out_filename)

    def set_paths(self):
        self.fast_align = os.path.join(self.fast_align_build, 'fast_align')
        self.atools = os.path.join(self.fast_align_build, 'atools')

    def read_err_files(self):
        (self.fwd_T, self.fwd_m) = self.read_err(self.fwd_err)
        (self.rev_T, self.rev_m) = self.read_err(self.rev_err)

    def read_err(self, err):
        (T, m) = ('', '')
        with open(err, mode = "r") as f:
            for line in f:
                # expected target length = source length * N
                if 'expected target length' in line:
                    m = line.split()[-1]
                # final tension: N
                elif 'final tension' in line:
                    T = line.split()[-1]
        return (T, m)

    def write_alignment_to_file(self, content, filename):
        with open(filename, mode = "w") as f:
            content = content.strip().split("\n")
            for line in content:
                line = line.strip().split('|||')
                if len(line) != 4:
                    print("Error: alignment failed")
                    print(line)
                    exit()
                f.write("{0}\n".format(line[2].strip()))

    def run_cmd(self, cmd):
        with subprocess.Popen(cmd, \
            stdin = None, \
            stdout = subprocess.PIPE, \
            stderr = subprocess.PIPE, \
            text = True, encoding = "utf-8",) as p:
            out, err = p.communicate()
        return out

    def run_alignment(self):
        fwd_cmd = [self.fast_align, '-i', self.corpus, '-d', '-T', self.fwd_T, '-m', self.fwd_m, '-f', self.fwd_params]
        fwd_out = self.run_cmd(fwd_cmd)
        rev_cmd = [self.fast_align, '-i', self.corpus, '-d', '-T', self.rev_T, '-m', self.rev_m, '-f', self.rev_params, '-r']
        rev_out = self.run_cmd(rev_cmd)
        self.write_alignment_to_file(fwd_out, self.fwd_out)
        self.write_alignment_to_file(rev_out, self.rev_out)
        atools_cmd = [self.atools, '-i', self.fwd_out, '-j', self.rev_out, '-c', self.heuristic]
        self.alignment = self.run_cmd(atools_cmd)

def main(args):
    aligner = Aligner(args.fast_align_build, args.corpus, args.fwd_params, args.fwd_err, \
        args.rev_params, args.rev_err, args.heuristic, args.out)
    aligner.run_alignment()
    aligner.write_result()

def more_help():
    print("""run:\n \
    \t fast_align -i corpus.f-e -d -v -o -p fwd_params >fwd_align 2>fwd_err\n \
    \t fast_align -i corpus.f-e -r -d -v -o -p rev_params >rev_align 2>rev_err\n \
    \n \
    then run:\n \
    \t {0} fwd_params fwd_err rev_params rev_err [heuristic] < in.f-e > out.f-e.gdfa \n \
    \n \
    where heuristic is one of: (intersect union grow-diag grow-diag-final grow-diag-final-and) default=grow-diag-final-and\n \
    """.format(sys.argv[0]))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Produce alignments from pre-trained model',)

    parser.add_argument('-he', '--heuristic', dest = 'heuristic', help = 'Heuristic (intersect, union grow-diag, grow-diag-final, grow-diag-final-and -- default: grow-diag-final-and)', required = False, default = 'grow-diag-final-and', type = str)

    arg_group = parser.add_argument_group('required arguments')
    arg_group.add_argument('-fa', '--fast-align-build', dest = 'fast_align_build', help = 'Path to fast_align build', required = True)
    arg_group.add_argument('-c', '--input-corpus', dest = 'corpus', help = 'Input corpus', required = True)
    arg_group.add_argument('-fp', '--forward-parameters', dest = 'fwd_params', help = 'Forward alignment parameters', required = True)
    arg_group.add_argument('-rp', '--reverse-parameters', dest = 'rev_params', help = 'Reverse alignment parameters', required = True)
    arg_group.add_argument('-fe', '--forward-error', dest = 'fwd_err', help = 'Forward alignment error', required = True)
    arg_group.add_argument('-re', '--reverse-error', dest = 'rev_err', help = 'Reverse alignment error', required = True)
    arg_group.add_argument('-o', '--output-file', dest = 'out', help = 'Output file', required = True)

    args = parser.parse_args()

    if len(vars(args)) >= 7:
        main(args)
    else:
        parser.print_help()
        more_help()
