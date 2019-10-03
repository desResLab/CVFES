#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Copyright, 2018
    Xue Li

    Main file of the CVFES project.
"""
from mpi4py import MPI
from cvfes import CVFES
from timeit import default_timer as timer
# Profiling.
import cProfile, pstats, io
import sys
import argparse

__author__ = "Xue Li"
__copyright__ = "Copyright 2018, the CVFES project"


def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Get the arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--samples', type=int, help='Number of samples.')
    parser.add_argument('-f', '--file', default='cfg/input.cfg', help='Configuration file.')

    args = parser.parse_args()

    cv = CVFES(comm)
    cv.ReadInputFile(args.file, nSmp=args.samples)

    if cv.Distribute() < 0:
        sys.exit(-1)

    if cv.Coloring() < 0:
        sys.exit(-1)

    start = timer()

    cv.Solve()

    end = timer()
    print('OK, rank: {} \t\t\t time: {:10.1f} ms'.format(rank, (end - start) * 1000.0))

    return rank

if __name__ == "__main__":

    # pr = cProfile.Profile()
    # pr.enable()

    rank = main()

    # pr.disable()

    # if rank == 0:
    #     s = io.StringIO()
    #     ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    #     ps.print_stats()
    #     print s.getvalue()

    #     # pr.dump_stats('profile/CVFES.prof')

