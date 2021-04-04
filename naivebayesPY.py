#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nigel
Modified by Wei shan Lee on Apr. 4, 2021.
"""

import numpy as np

def naivebayesPY(x, y):
    """
    naivebayesPY(Y) returns [pos,neg]

    Computation of P(Y)
    Input:
        X : n input vectors of d dimensions (nxd)
        Y : n labels (-1 or +1) (nx1)

    Output:
        pos: probability p(y=1)
        neg: probability p(y=-1)
    """
    
    # add one positive and negative example to avoid division by zero ("plus-one smoothing")
    Y = np.concatenate([Y, [-1,1]])
    n = len(Y)
    
    pos = np.count_nonzero(Y == 1) / n
    neg = np.count_nonzero(Y == -1) / n
    return pos,neg
