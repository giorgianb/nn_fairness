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
import quad
from prob import ProbabilityDensityComputer

from nnenum.network import NeuralNetwork
from typing import Union

from collections import namedtuple
from collections.abc import Mapping, Sequence


integrate = quad.block_qhull

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

def get_bounding_box(lpi):
    bb = []
    A = lpi.get_constraints_csr().toarray()
    b = lpi.get_rhs()
    for i in range(A.shape[1]):
        v = np.zeros(A.shape[1])
        v[i] = 1
        min_bound = lpi.minimize(v)[i]
        max_bound = lpi.minimize(-v)[i]
        bb.append((min_bound, max_bound))

    return np.array(bb)

def bounding_boxes_overlap(bb0, bb1):
    return not (np.any(bb0[1, :] < bb1[0, :]) or np.any(bb1[1, :] < bb0[0, :]))

def load_class_prob(config):
    with open(config['train_data_path'], 'rb') as f:
        data_dict = pickle.load(f)
        X = data_dict['X_train']

    c1 = config['class_1']
    c2 = config['class_2']
    class_1_indices = np.array(c1['indices'])
    class_1_values = np.array(c1['values'])
    class_2_indices = np.array(c2['indices'])
    class_2_values = np.array(c2['values'])
    def is_class_1(x):
        return np.allclose(x[class_1_indices], class_1_values, atol=1e-1)

    def is_class_2(x):
        return np.allclose(x[class_2_indices], class_2_values, atol=1e-1)

    class_1_prob = ProbabilityDensityComputer(
            X,
            config['discrete_indices'],
            config['continuous_indices'],
            config['one_hot_indices'],
            config['fixed_indices'],
            is_class_1
    )

    class_2_prob = ProbabilityDensityComputer(
            X,
            config['discrete_indices'],
            config['continuous_indices'],
            config['one_hot_indices'],
            config['fixed_indices'],
            is_class_2
    )

    return class_1_prob, class_2_prob

def run_on_model(config, model_index):
    set_settings()
    with open(config['train_data_path'], 'rb') as f:
        data_dict = pickle.load(f)
        X = data_dict['X_train']

    c1 = config['class_1']
    c2 = config['class_2']
    class_1_indices = np.array(c1['indices'])
    class_1_values = np.array(c1['values'])
    class_2_indices = np.array(c2['indices'])
    class_2_values = np.array(c2['values'])
    def is_class_1(x):
        return np.allclose(x[class_1_indices], class_1_values, atol=1e-1)

    def is_class_2(x):
        return np.allclose(x[class_2_indices], class_2_values, atol=1e-1)

    class_1_prob = ProbabilityDensityComputer(
            X,
            config['discrete_indices'],
            config['continuous_indices'],
            config['one_hot_indices'],
            config['fixed_indices'],
            is_class_1
    )

    class_2_prob = ProbabilityDensityComputer(
            X,
            config['discrete_indices'],
            config['continuous_indices'],
            config['one_hot_indices'],
            config['fixed_indices'],
            is_class_2
    )

    class_1_box = c1['box']
    class_2_box = c2['box']

    # results_dict['model_size']['metric']['fairness_action']
    results_dict = defaultdict(lambda: defaultdict(dict))
    network_label, onnx_filename, auc = config['models'][model_index]
    network = load_onnx_network_optimized(onnx_filename)

    inits = [class_1_box, class_2_box] #[male_inits, female_inits]
    probs = [class_1_prob, class_2_prob]
    labels = [c1['label'], c2['label']]

    lpi_polys = []
    total_probabilities = []
    total_time = 0
    for i, (init, prob, label) in enumerate(zip(inits, probs, labels)):
        lpi_polys.append([])


        for hot_indices in product(*config['one_hot_indices']):
            init_box = np.array(init, dtype=np.float32)
            init_box[list(hot_indices)] = 1
            t1 = time.perf_counter()
            res = enumerate_network(init_box, network)
            t2 = time.perf_counter()
            total_time += (t2 - t1)
            result_str = res.result_str
            assert result_str == "none"

            print(f"[{(network_label, model_index)}] {labels[i]} split into {len(res.stars)} polys")

            for star in res.stars:
                # add constaint that output < 0 (low risk)
                assert star.a_mat.shape[0] == 1, "single output should mean single row"
                row = star.a_mat[0]
                bias = star.bias[0]

                star.lpi.add_dense_row(row, -bias)
                #star.lpi.add_dense_row(-2*row, -bias - 1)

                if star.lpi.is_feasible():
                    bounding_box = get_bounding_box(star.lpi)
                    lpi_polys[i].append((star.lpi, bounding_box))



    print(f"[{(network_label, model_index)}] lp_polys size: {tuple(len(poly) for poly in lpi_polys)}") 
    try:
        for label_0, polys_0, prob_0 in zip(labels, lpi_polys, probs):
            total_probability = 0
            print(f"[Calculating total probability]")
            for lpi, bounding_box in tqdm.tqdm(polys_0):
                total_probability += integrate(lpi, prob_0, config['fixed_indices'])

            print("total probability:", total_probability)
            for label_1, polys_1, prob_1 in zip(labels, lpi_polys, probs):
                if label_0 == label_1:
                    continue

                pref_prob = 0
                print(f"[Calculating Preference Probability]")
                for lpi, bounding_box in tqdm.tqdm(polys_1):
                    pref_prob += integrate(lpi, prob_1, config['fixed_indices'])
                print("preference probability:", pref_prob)

                adv_prob = 0
                print(f"[Calculating Advantage Probability]")
                for (lpi_0, bb_0), (lpi_1, bb_1) in tqdm.tqdm(tuple(product(polys_0, polys_1))):
                    if not bounding_boxes_overlap(bb_0, bb_1):
                        pass
                        #continue

                    intersection_lpi = compute_intersection_lpi(lpi_0, lpi_1)

                    if intersection_lpi.is_feasible():
                        adv_prob += integrate(intersection_lpi, prob_0, config['fixed_indices'])

                print("advantage probability:", adv_prob)
                results_dict[network_label]['Advantage'][f"{label_0},{label_1}"] = total_probability - adv_prob
                results_dict[network_label]['Preference'][f"{label_0},{label_1}"] = total_probability - pref_prob

                print(f"[{(model_index, network_label)}] {label_0} advantage over {label_1}: {total_probability - adv_prob}")
                print(f"[{(model_index, network_label)}] {label_0} preference over {label_1}: {total_probability - pref_prob}")
        results_dict[network_label]['Symmetric Difference'] = sum(results_dict[network_label]['Advantage'].values())
        results_dict[network_label]['Net Preference'] = sum(map(lambda x: max(x, 0), results_dict[network_label]['Advantage'].values()))
        results_dict[network_label]['AUC'] = auc
    except Exception:
        raise

    return results_dict




def main():
    """main entry point"""

    init_plot()


    # ideas:
    # Initial set defined as star: (triangle), where unused input dimension is the pdf
    #
    # then, to integrate you just compute the area at the end
    #
    # symmetric difference = area1 + area2 - 2*area of intersection
    # area of intersection can be optimized using zonotope box bounds
    with open(sys.argv[1], 'r') as handle:
        config = json.load(handle)


    n_models = len(config['models'])
    results = Parallel(n_jobs=8)(delayed(run_on_model)(config, i) for i in range(n_models))

    results_dict = {}
    for result in results:
        for k, v in result.items():
            results_dict[k] = v

    with open(sys.argv[2], 'w') as handle:
        print(f"Saving result to: {sys.argv[2]}")
        json.dump(results_dict, handle)


if __name__ == "__main__":
    main()

InputConfig = namedtuple("InputConfig", ("one_hot_indices", "discrete_indices"))

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
                    row = star.a_mat[0]
                    bias = star.bias[0]

                    # TODO: look into why we do this
                    star.lpi.add_dense_row(row, -bias)
                    #star.lpi.add_dense_row(-2*row, -bias - 1)

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
        fixed_indices = []
        for i, (l, u) in enumerate(init_box):
            if l == u:
                fixed_indices.append(i)
        return fixed_indices

class FairnessMetrics:
    def __init__(self, lpi_polys: Mapping, fixed_indices: Mapping):
        self.lpi_polys = lpi_polys
        self.fixed_indices = dict(fixed_indices)

    # Advantage: percentage of whites not accepted as blacks
    def advantage(self, probs: Union[Mapping, Sequence], label=None):
        if isinstance(probs, Sequence):
            probs_dict = {}
            for i, prob in enumerate(probs):
                probs_dict[i] = prob
            probs = probs_dict

        advantage = {}
        for label_0, prob_0 in probs.items():
            total_probability = 0
            polys_0 = self.lpi_polys[label_0]

            for lpi, bounding_box in polys_0:
                total_probability += integrate(lpi, prob_0, self.fixed_indices[label_0])

            for label_1, prob_1 in probs.items():
                if label_0 == label_1:
                    continue

                polys_1 = self.lpi_polys[label_1]
                adv_prob = 0
                for (lpi_0, bb_0), (lpi_1, bb_1) in product(polys_0, polys_1):
                    # TODO: fix. This does not work correctly!!
                    #if not bounding_boxes_overlap(bb_0, bb_1):
                    #    continue

                    intersection_lpi = compute_intersection_lpi(lpi_0, lpi_1)

                    if intersection_lpi.is_feasible():
                        adv_prob += integrate(intersection_lpi, prob_0, self.fixed_indices[label_0])
                if label is None:
                    result = (total_probability - adv_prob, total_probability)
                else:
                    diff = total_probability - adv_prob
                    result = (total_probability - adv_prob, total_probability, f'{diff*100}% of all {label_0} individuals would have not been classified {label} if only they had been {label_1}')
                advantage[(label_0, label_1)] = result

        return advantage

    # Disadvantage: Proportion of blacks that would have been accepted as white
    def disadvantage(self, probs: Union[Mapping, Sequence], label=None):
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
                for lpi, bounding_box in polys_0:
                    counterfactual_probability += integrate(lpi, prob_1, self.fixed_indices[label_1])


                polys_1 = self.lpi_polys[label_1]
                disadv_prob = 0
                # We calculate how many are accepted under both rules
                for (lpi_0, bb_0), (lpi_1, bb_1) in product(polys_0, polys_1):
                    #if not bounding_boxes_overlap(bb_0, bb_1):
                    #    continue

                    intersection_lpi = compute_intersection_lpi(lpi_0, lpi_1)

                    if intersection_lpi.is_feasible():
                        disadv_prob += integrate(intersection_lpi, prob_1, self.fixed_indices[label_1])
                if label is None:
                    result = (counterfactual_probability - disadv_prob, counterfactual_probability)
                else:
                    diff = counterfactual_probability - disadv_prob
                    result = (counterfactual_probability - disadv_prob, counterfactual_probability, f'{diff*100}% of all {label_1} individuals would have been classified {label} if only they had been {label_0}')

                disadvantage[(label_1, label_0)] = result

        return disadvantage

    def preference(self, probs: Union[Mapping, Sequence], label=None):
        if isinstance(probs, Sequence):
            probs_dict = {}
            for i, prob in enumerate(probs):
                probs_dict[i] = prob
            probs = probs_dict

        preference = {}
        for label_0, prob_0 in probs.items():
            polys_0 = self.lpi_polys[label_0]
            total_probability = 0

            for lpi, bounding_box in polys_0:
                total_probability += integrate(lpi, prob_0, self.fixed_indices[label_0])

            for label_1, prob_1 in probs.items():
                if label_0 == label_1:
                    continue

                polys_1 = self.lpi_polys[label_1]
                pref_prob = 0
                for lpi, bounding_box in polys_1:
                    pref_prob += integrate(lpi, prob_1, self.fixed_indices[label_1])

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
