#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Copyright, 2018
    Xue Li

    Calculate the parobolic velocity profile.
"""

import numpy as np
from optimizedParabolicVelocityProfile import OptimizedParabolicVelocityProfile


def ParabolicVelocityProfile(elements, nodes, glbNodeIds, boundary):
    """ Calculate the parabolic velocity profile for the inlet.
        - elements: elements on the inlet
        - nodes: global nodes information
        - glbNodeIds: the corresponding global position for each node

        return: u, int_u
    """
    
    nNodes = glbNodeIds.shape[0]
    K = np.zeros((nNodes, nNodes))
    f = np.zeros(nNodes)
    Ae = np.zeros(elements.shape[0])

    OptimizedParabolicVelocityProfile(nodes, elements, glbNodeIds, K, f, Ae)

    # Apply zero velocity at the boundary.
    for i in boundary:
        K[i,:] = 0.0
        K[i,i] = 1.0
        f[i] = 0.0
        
    u = np.linalg.solve(K, f)
    int_u = np.sum(np.mean(u[elements], axis=1) * Ae)

    return u, int_u