# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 13:55:37 2026

@author: samth
"""
raise
import itertools
import pandas as pd
import pickle as pk
import pathlib
import numpy as np
import time
import re
import os
from copy import deepcopy
libdir = pathlib.Path(__file__).resolve().with_name('modules')
computerSpecificPath = str(pathlib.Path(*list(libdir.parts)[:np.where(pd.Series(libdir.parts).str.contains('OneDrive'))[0][0]+1])) + "\\" ## This finds the path to onedrive
computerSpecificPath_pl = pathlib.Path(computerSpecificPath)
# raise
with open(computerSpecificPath_pl.joinpath(r"WS_DL\Lab Data\Price\code\filePathsDictionary.pkl"), 'rb') as fh:
    filePathsDictionary = pk.load(fh)
    

for filePathsDictionary_key in filePathsDictionary.keys():
    if r"C:\Users\sfe8458\OneDrive - Northwestern University\WS_DL\Lab Data\Price" in filePathsDictionary[filePathsDictionary_key]:
        filePathsDictionary.update({filePathsDictionary_key:filePathsDictionary[filePathsDictionary_key].replace(r"C:\Users\sfe8458\OneDrive - Northwestern University\WS_DL\Lab Data\Price", str(computerSpecificPath_pl.joinpath(r"WS_DL\Lab Data\Price")))})

from importlib.machinery import SourceFileLoader 

projOntoBufferedInteriorSimplex_flexibleDir = SourceFileLoader('projOntoBufferedInteriorSimplex_flexibleDir', filePathsDictionary['projOntoBufferedInteriorSimplex_flexibleDir']).load_module()

bufferedProj_input = projOntoBufferedInteriorSimplex_flexibleDir.projOntoBufferedInteriorSimplex(1e-4)

bufferedProj_input.precomputeForSetDimension(ndims=3)

import pickle as pk
with open(r"C:\Users\samth\OneDrive - Northwestern University\WS_DL\Lab Data\Price\code\fullSystemData\allValid_3Element_compositions_0.01inc.pkl", 'rb') as fh:
    ternComps_arr = pk.load(fh)
    
    
def compositions(arr, n, target=1.0, tol=1e-12):
    arr = sorted(arr)

    def search(prefix, remaining, total):
        if remaining == 0:
            if abs(total - target) <= tol:
                yield tuple(prefix)
            return

        remaining_target = target - total

        # Prune: even if we choose the smallest/largest possible
        # value for every remaining position, can we still reach target?
        if remaining_target < remaining * arr[0] - tol:
            return

        if remaining_target > remaining * arr[-1] + tol:
            return

        for x in arr:
            if x > remaining_target + tol:
                break

            yield from search(
                prefix + [x],
                remaining - 1,
                total + x
            )

    yield from search([], n, 0.0)


ternComps_arr = ternComps_arr[np.linalg.norm((ternComps_arr*100/2)-np.round(ternComps_arr*100/2), axis=1)<1e-9].copy()

ternComps_alt_arr = np.array(list(compositions(arr=np.round(np.arange(0, 1+1e-10, 0.02), 8).copy(), n=3, target=1.0, tol=1e-4))).copy()
(ternComps_alt_arr == ternComps_arr).all(axis=1).all()
# ternComps_alt_arr[np.invert((ternComps_alt_arr == ternComps_arr).all(axis=1))][0,2]
# ternComps_arr[np.invert((ternComps_alt_arr == ternComps_arr).all(axis=1))][0,2]

projed_arr = np.array([bufferedProj_input.proj(row) for row in ternComps_arr]).copy()
projed_arr = np.round(projed_arr, 10).copy()
(projed_arr == ternComps_arr).all(axis=1).sum()
len(ternComps_arr) - (projed_arr == ternComps_arr).all(axis=1).sum()
(ternComps_arr==0).any(axis=1).sum()
np.linalg.norm((projed_arr - ternComps_arr), axis=1)

# with open(r"C:\Users\samth\OneDrive - Northwestern University\WS_DL\Lab Data\Price\code\kawin\examples\ternaryExamples\allValid_3Element_compositions_0.01inc_projTo1eminus4.pkl", 'wb') as fh:
#     pk.dump(projed_arr, fh)

# with open(r"C:\Users\samth\OneDrive - Northwestern University\WS_DL\Lab Data\Price\code\kawin\examples\ternaryExamples\allValid_3Element_compositions_0.02inc_projTo1eminus4.pkl", 'wb') as fh:
#     pk.dump(projed_arr, fh)

# with open(r"C:\Users\samth\OneDrive - Northwestern University\WS_DL\Lab Data\Price\code\kawin\examples\ternaryExamples\allValid_3Element_compositions_0.01inc_projTo1eminus4.pkl", 'rb') as fh:
#     projed_arr = pk.load(fh)
