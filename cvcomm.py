#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Copyright, 2018
    Xue Li

    CVConfig is the configuration structure of the CVFES project.
"""

from mpi4py import MPI

__author__ = "Xue Li"
__copyright__ = "Copyright 2018, the CVFES project"


class CVCOMM:

    def __init__(self, comm):
        self.comm = comm
        self.size = comm.Get_size() # Number of proces in comm group
        self.rank = comm.Get_rank()
