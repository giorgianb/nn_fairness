import numpy as np
import time

import matplotlib
import matplotlib.pyplot as plt

from scipy.integrate import quad

from nnenum.enumerate import enumerate_network
from nnenum.settings import Settings
from nnenum.result import Result
from nnenum.onnx_network import load_onnx_network_optimized, load_onnx_network
from nnenum import kamenev
from nnenum.lpinstance import LpInstance
from scipy.optimize import linprog

import glpk_util
from icecream import ic
import os.path
import pickle
import tqdm
from scipy.sparse import csr_matrix
from itertools import product
import csv
import copy
import json

from collections import defaultdict

import swiglpk as glpk
import sys
from joblib import Parallel, delayed

from nnenum.network import NeuralNetwork
from typing import Union

from collections import namedtuple
from collections.abc import Mapping, Sequence

from rtree import index
from polytope import Polytope, reduce
from sklearn.decomposition import PCA


def set_settings():
    """exact analysis settings"""

    Settings.PRINT_OUTPUT = False
    Settings.TIMING_STATS = False
    Settings.TRY_QUICK_OVERAPPROX = False

    Settings.CONTRACT_ZONOTOPE_LP = True
    Settings.CONTRACT_LP_OPTIMIZED = True
    Settings.CONTRACT_LP_TRACK_WITNESSES = True

    Settings.OVERAPPROX_BOTH_BOUNDS = False

    Settings.BRANCH_MODE = Settings.BRANCH_EXACT

    Settings.RESULT_SAVE_STARS = True
    Settings.NUM_PROCESSES = 1 # single-threaded, easier to debug

def init_plot():
    'initialize plotting style'

    #matplotlib.use('TkAgg') # set backend
    plt.style.use(['bmh', 'bak_matplotlib.mlpstyle'])


def compute_intersection_lpi(lpi1, lpi2):
    """compute the intersection between two lpis"""

    # sanity checks
    assert lpi1.get_num_cols() == lpi2.get_num_cols()

    cols = lpi1.get_num_cols()

    assert np.allclose(lpi1._get_col_bounds(), lpi2._get_col_bounds())

    rv = LpInstance(lpi1)
    # add constraints of lpi 2 to rv

    rhs = lpi2.get_rhs()
    mat = lpi2.get_constraints_csr().toarray()

    for row, val in zip(mat, rhs):
        rv.add_dense_row(row, val)

    return rv

def get_bounding_box(lpi, directions=None):
    bb = []
    if directions is None:
        A = lpi.get_constraints_csr().toarray()
        directions = np.eye(A.shape[1])
    for i, v in enumerate(directions):
        min_bound = (v @ lpi.minimize(v))
        max_bound = (v @ lpi.minimize(-v))
        bb.append((min_bound, max_bound))

    return np.array(bb)

class FairnessCalculator:
    def __init__(self, network: NeuralNetwork):
        self.network = network

    @staticmethod
    def from_onnx_file(filename: str):
        return FairnessCalculator(load_onnx_network_optimized(filename))

    def __call__(self, classes: Union[Mapping, Sequence]):
        if isinstance(classes, Sequence):
            classes_dict = {}
            for i, init_region in enumerate(classes):
                classes_dict[i] = init_region
            classes = classes_dict

        lpi_polys = {}
        fixed_indices = {}
        for class_label, boxes in classes.items():
            lpi_polys[class_label] = []
            for init_box in boxes:
                fixed_indices[class_label] = self._compute_fixed_indices(init_box)
                res = enumerate_network(init_box, self.network)
                for star in res.stars:
                    #print(f'{star.a_mat.shape=}, {star.bias.shape=}')
                    #print(f'{star.a_mat=}')
                    #print(f'{star.bias=}')
                    row = star.a_mat[0]
                    bias = star.bias[0]
                    A = star.lpi.get_constraints_csr().toarray()
                    b = star.lpi.get_rhs()
                    #print(f'{A=} {b=}')
                    #print(f'{star.a_mat=} {star.bias=}')
                    #print('lpi: ', star.lpi)

                    # TODO: look into why we do this
                    #star.lpi.add_dense_row(row, -bias)
                    star.lpi.add_dense_row(row, -bias)
                    #star.lpi.add_dense_row(-2*row, -bias - 1)
                    #print('lpi after: ', star.lpi)

                    if star.lpi.is_feasible():
                        bounding_box = get_bounding_box(star.lpi)
                        lpi_polys[class_label].append((star.lpi, bounding_box))

        return FairnessMetrics(lpi_polys, fixed_indices)

    @staticmethod
    def _fix_hot_indices(init_box, hot_indices: Sequence[int]):
        fixed_box = init_box.copy()
        for i, hot_index in enumerate(hot_indices):
            fixed_box[i] = hot_index
        return fixed_box

    def _compute_fixed_indices(self, init_box):
        return np.where(init_box[:, 0] == init_box[:, 1])[0]

class FairnessMetrics:
    def __init__(self, lpi_polys: Mapping, fixed_indices: Mapping):
        self.lpi_polys = lpi_polys
        self.fixed_indices = dict(fixed_indices)
        self.intervals = {}
        for k, v in lpi_polys.items():
            if len(v) == 0:
                continue

            n = v[0][1].shape[0]
            p = index.Property(dimension=n)
            self.intervals[k] = index.Index(interleaved=False, properties=p)
            for i, (lpi, bb) in enumerate(v):
                self.intervals[k].insert(i, bb.flatten())

    # Advantage: percentage of whites not accepted as blacks
    def advantage(self, probs: Union[Mapping, Sequence], label=None, progress=False):
        def progress_bar(item, total=None, desc=None):
            if progress:
                if total is None and hasattr(item, '__len__'):
                    total = len(item)
                return tqdm.tqdm(item, total=total, desc=desc)
            else:
                return item

        if isinstance(probs, Sequence):
            probs_dict = {}
            for i, prob in enumerate(probs):
                probs_dict[i] = prob
            probs = probs_dict

        advantage = {}
        for label_0, prob_0 in probs.items():
            total_probability = 0
            polys_0 = self.lpi_polys[label_0]


            for lpi, bounding_box in progress_bar(polys_0):
                total_probability += prob_0.integrate(lpi, self.fixed_indices[label_0])

            for label_1, prob_1 in probs.items():
                if label_0 == label_1:
                    continue

                polys_1 = self.lpi_polys[label_1]
                adv_prob = 0

                intervals = self.intervals[label_1]
                for i, (lpi_0, bb_0) in enumerate(progress_bar(polys_0)):
                    candidates = intervals.intersection(bb_0.flatten())
                    for candidate in candidates:
                        lpi_1, _ = polys_1[candidate]
                        intersection_lpi = compute_intersection_lpi(lpi_0, lpi_1)

                        if intersection_lpi.is_feasible():
                            adv_prob += prob_0.integrate(intersection_lpi, self.fixed_indices[label_0])


                if label is None:
                    result = (max(total_probability - adv_prob, 0), total_probability)
                else:
                    diff = max(total_probability - adv_prob, 0)
                    result = (max(total_probability - adv_prob, 0), total_probability, f'{diff*100}% of all {label_0} individuals would have not been classified {label} if only they had been {label_1}')
                advantage[(label_0, label_1)] = result

        return advantage

    # Disadvantage: Proportion of blacks that would have been accepted as white
    def disadvantage(self, probs: Union[Mapping, Sequence], label=None, progress=False):
        def progress_bar(item, total=None, desc=None):
            if progress:
                if total is None and hasattr(item, '__len__'):
                    total = len(item)
                return tqdm.tqdm(item, total=total, desc=desc)
            else:
                return item


        if isinstance(probs, Sequence):
            probs_dict = {}
            for i, prob in enumerate(probs):
                probs_dict[i] = prob
            probs = probs_dict

        counterfactual_probabilities = {}
        disadvantage = {}


        for label_0, prob_0 in probs.items():
            polys_0 = self.lpi_polys[label_0]

            for label_1, prob_1 in probs.items():
                if label_0 == label_1:
                    continue

                # We calculate the acceptance proportion evaluating it using the rules of the other race
                counterfactual_probability = 0
                for lpi, bounding_box in progress_bar(polys_0):
                    counterfactual_probability += prob_1.integrate(lpi, self.fixed_indices[label_1])


                polys_1 = self.lpi_polys[label_1]
                disadv_prob = 0

                # We calculate how many are accepted under both rules
                intervals = self.intervals[label_1]
                for lpi_0, bb_0 in progress_bar(polys_0):
                    candidates = intervals.intersection(bb_0.flatten())
                    for candidate in candidates:
                        lpi_1, _ = polys_1[candidate]
                        intersection_lpi = compute_intersection_lpi(lpi_0, lpi_1)

                        if intersection_lpi.is_feasible():
                            disadv_prob += prob_1.integrate(intersection_lpi, self.fixed_indices[label_1])

                if label is None:
                    result = (max(counterfactual_probability - disadv_prob, 0), counterfactual_probability)
                else:
                    diff = max(counterfactual_probability - disadv_prob, 0)
                    result = (max(counterfactual_probability - disadv_prob, 0), counterfactual_probability, f'{diff*100}% of all {label_1} individuals would have been classified {label} if only they had been {label_0}')

                disadvantage[(label_1, label_0)] = result

        return disadvantage

    def preference(self, probs: Union[Mapping, Sequence], label=None, progress=False):
        def progress_bar(item, total=None, desc=None):
            if progress:
                if total is None and hasattr(item, '__len__'):
                    total = len(item)
                return tqdm.tqdm(item, total=total, desc=desc)
            else:
                return item

        if isinstance(probs, Sequence):
            probs_dict = {}
            for i, prob in enumerate(probs):
                probs_dict[i] = prob
            probs = probs_dict

        preference = {}
        for label_0, prob_0 in probs.items():
            polys_0 = self.lpi_polys[label_0]
            total_probability = 0

            for lpi, _ in progress_bar(polys_0):
                total_probability += prob_0.integrate(lpi, self.fixed_indices[label_0])

            for label_1, prob_1 in probs.items():
                if label_0 == label_1:
                    continue

                polys_1 = self.lpi_polys[label_1]
                pref_prob = 0
                for lpi, _ in progress_bar(polys_1):
                    pref_prob += prob_1.integrate(lpi, self.fixed_indices[label_1])

                pref_prob = max(pref_prob, 0)
                if total_probability == 0:
                    if pref_prob == 0:
                        result = (1, total_probability)
                    else:
                        result = (np.inf, total_probability)
                else:
                    result = (pref_prob, total_probability)

                if label is not None:
                    result = (*result, f'{label_1} individuals are considered {label} at {result[0]/result[1]*100}% of the rate of {label_0} individuals')
                preference[(label_1, label_0)] = result

        return preference
