#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Copyright, 2018
    Xue Li

    Main file of the CVFES project.
"""
from mpi4py import MPI
from cvcomm import CVCOMM
from cvfes import CVFES

__author__ = "Xue Li"
__copyright__ = "Copyright 2018, the CVFES project"


def main():

    # Construct comunications
    # cvComm = CVCOMM(MPI.COMM_WORLD)
    comm = MPI.COMM_WORLD

    cv = CVFES(comm)

    cv.ReadInputFile("input.cfg")

    cv.Distribute(0)

    cv.Solve(0)

if __name__ == "__main__":
    main()
