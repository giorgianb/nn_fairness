"""
fairness analyzer exploration using nnenum
Nov 2021, Stanley Bak
"""

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from scipy.integrate import quad

from nnenum.enumerate import enumerate_network
from nnenum.settings import Settings
from nnenum.result import Result
from nnenum.onnx_network import load_onnx_network_optimized, load_onnx_network
from nnenum import kamenev
from nnenum.lpinstance import LpInstance

import glpk_util
from compute_volume import quad_integrate_glpk_lp, rand_integrate_polytope, quad_integrate_polytope, qhull_integrate_polytope


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
from itertools import chain

INTEGRATION = 'block-qhull'

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

def integrate(lpi, pdf):
    global INTEGRATION
    if INTEGRATION == 'random':
        A = lpi.get_constraints_csr().toarray()
        b = lpi.get_rhs()
        return rand_integrate_polytope(A, b, func_to_integrate=pdf.sample, samples=100000)
    elif INTEGRATION == 'quad':
        return quad_integrate_glpk_lp(lpi.lp, pdf.sample)
    elif INTEGRATION == 'block':
        prob = 0
        for region in pdf.regions:
            region = np.array(region)
            A = lpi.get_constraints_csr().toarray()
            b = lpi.get_rhs()
            point = []
            for i, (lbound, ubound) in enumerate(region):
                point.append((lbound + ubound) / 2)
                row = np.zeros((1, len(region)))
                row[0, i] = 1
                A = np.concatenate((A, row, -row), axis=0)
                b = np.append(b, (ubound, -lbound))
            prob += quad_integrate_polytope(A, b)*pdf.sample(*point)

        return prob
    elif INTEGRATION == 'block-qhull':
        prob = 0

        A_lpi = lpi.get_constraints_csr().toarray()
        b_lpi = lpi.get_rhs()
        lpi_copy = LpInstance(lpi)
       
        for region in pdf.regions:
            region = np.array(region)

            A = A_lpi.copy()
            b = b_lpi.copy()
            # check if it's feasible before computing volume
            for i, (lbound, ubound) in enumerate(region):
                if lbound == ubound:
                    glpk.glp_set_col_bnds(lpi_copy.lp, i + 1, glpk.GLP_FX, lbound, lbound) # needs: import swiglpk as glpk
                else:
                    glpk.glp_set_col_bnds(lpi_copy.lp, i + 1, glpk.GLP_DB, lbound, ubound) # needs: import swiglpk as glpk

            feasible = lpi_copy.is_feasible()
            if not feasible:
                continue

            point = []
            to_eliminate_cols = []
            to_eliminate_vals = []
            to_keep_cols = []
            for i, (lbound, ubound) in enumerate(region):
                p = lbound if lbound == ubound else (lbound + ubound) / 2
                point.append(p)
                if lbound == ubound:
                    to_eliminate_cols.append(i)
                    to_eliminate_vals.append(lbound)
                else:
                    row = np.zeros((1, len(region)))
                    row[0, i] = 1
                    A = np.concatenate((A, row, -row), axis=0)
                    b = np.append(b, (ubound, -lbound))
                    to_keep_cols.append(i)
               
            p = pdf.sample(*point)
            if p == 0:
                continue

            if len(to_eliminate_cols) > 0:
                to_eliminate_vals = np.array(to_eliminate_vals)
                b -= A[:, to_eliminate_cols] @ to_eliminate_vals
                A = A[:, to_keep_cols]

            prob += qhull_integrate_polytope(A, b)*p
            #prob += quad_integrate_polytope(A, b)*pdf.sample(*point)

        return prob
    elif INTEGRATION == 'block-linear':
        prob = 0
        for region in pdf.regions:
            region = np.array(region)
            A = lpi.get_constraints_csr().toarray()
            b = lpi.get_rhs()
            for i, (lbound, ubound) in enumerate(region):
                row = np.zeros((1, len(region)))
                row[0, i] = 1
                A = np.concatenate((A, row, -row), axis=0)
                b = np.append(b, (ubound, -lbound))
            prob += quad_integrate_polytope(A, b, pdf.sample)

        return prob



def make_linear_interpolation_func(pts):
    """converts a list of 2-d points to an interpolation function
    assumes function is zero outside defined range
    """

    assert len(pts) > 1

    last_x = pts[0][0]
    
    for x, _ in pts[1:]:
        assert x > last_x, "first argument in pts must be strictly increasing"
        last_x = x

    def f(x):
        """the linear interpolation function"""

        assert isinstance(x, (int, float)), f"x was {type(x)}"

        if x < pts[0][0] or x > pts[-1][0]:
            rv = 0
        else:
            # binary search
            a = 0
            b = len(pts) - 1

            while a + 1 != b:
                mid = (a + b) // 2

                if x < pts[mid][0]:
                    b = mid
                else:
                    a = mid

            # at this point, interpolate between a and b
            a_arg = pts[a][0]
            b_arg = pts[b][0]
            
            ratio = (x - a_arg) / (b_arg - a_arg) # 0=a, 1=b
            assert 0 <= ratio <= 1

            val_a = pts[a][1]
            val_b = pts[b][1]
            
            rv = (1-ratio)*val_a + ratio*val_b

        return rv

    return f

def make_continuous_distribution(data):
    counts, boundaries = np.histogram(data, bins=10)
    centers = (boundaries[1:] + boundaries[:-1])/2
    distribution = np.stack((centers, counts), axis=-1)

    return distribution

def make_discrete_distribution(data):
    dist = defaultdict(int)

    for x in data:
        dist[x] += 1

    return dist

def make_discrete_func(dist):
    def func(x):
        return dist.get(x, 0)

    return func



class ProbabilityDensityComputer:
    """computes probability of input at given point"""

    def __init__(self, X, discrete_indices, continuous_indices, class_filter):
        matches_given = np.apply_along_axis(class_filter, 1, X)
        X = X[matches_given]

        self.discrete_indices = tuple(discrete_indices)
        self.continuous_indices = tuple(continuous_indices)
        self.continuous = [make_continuous_distribution(X[:, i]) for i in continuous_indices]
        self.discrete = [make_discrete_distribution(X[:, i]) for i in discrete_indices]
        self.continuous_funcs = [make_linear_interpolation_func(d) for d in self.continuous]
        self.discrete_funcs = [make_discrete_func(d) for d in self.discrete]
        self.continuous_volumes = [quad(f, d[0][0], d[-1][0], limit=500)[0] for f, d in zip(self.continuous_funcs, self.continuous)]
        self.discrete_volumes = [sum(f(k) for k in d.keys()) for f, d in zip(self.discrete_funcs, self.discrete)]
        funcs = sorted(
                chain(
                    zip(continuous_indices, self.continuous_funcs, self.continuous_volumes),
                    zip(discrete_indices, self.discrete_funcs, self.discrete_volumes),
                )
        )
        self.funcs = tuple(map(lambda x: (x[1], x[2]), funcs))
        self.continuous_bounds = [tuple(zip(d[:, 0], d[1:, 0])) for d in self.continuous]
        self.discrete_bounds = [[(k, k) for k in sorted(d.keys())] for d in self.discrete]
        regions = sorted(
                chain(
                    zip(continuous_indices, self.continuous_bounds),
                    zip(discrete_indices, self.discrete_bounds)
                )
        )
        self._regions = tuple(map(lambda x: x[1], regions))



    def sample(self, *args):
        """get probability density at a point"""

        p = 1
        for (f, v), x in zip(self.funcs, args):
            p *= f(x)/v

        return p

    @property
    def regions(self):
        return product(*self._regions)


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

def main():
    """main entry point"""

    init_plot()
    set_settings()


    # ideas:
    # Initial set defined as star: (triangle), where unused input dimension is the pdf
    #
    # then, to integrate you just compute the area at the end
    #
    # symmetric difference = area1 + area2 - 2*area of intersection
    # area of intersection can be optimized using zonotope box bounds
    with open(sys.argv[1], 'r') as handle:
        config = json.load(handle)

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
        return np.all(x[class_1_indices] == class_1_values)

    def is_class_2(x):
        return np.all(x[class_2_indices] == class_2_values)

    class_1_prob = ProbabilityDensityComputer(
            X,
            config['discrete_indices'],
            config['continuous_indices'],
            is_class_1
    )

    class_2_prob = ProbabilityDensityComputer(
            X,
            config['discrete_indices'],
            config['continuous_indices'],
            is_class_2
    )

    class_1_box = c1['box']
    class_2_box = c2['box']

    # results_dict['model_size']['metric']['fairness_action']
    results_dict = defaultdict(lambda: defaultdict(dict))
    for network_label, onnx_filename, auc in config['models']:
        network = load_onnx_network_optimized(onnx_filename)

        inits = [class_1_box, class_2_box] #[male_inits, female_inits]
        probs = [class_1_prob, class_2_prob]
        labels = [c1['label'], c2['label']]

        lpi_polys = []
        total_probabilities = []
        for i, (init, prob, label) in enumerate(zip(inits, probs, labels)):
            lpi_polys.append([])

            init_box = np.array(init, dtype=np.float32)

            res = enumerate_network(init_box, network)
            result_str = res.result_str
            assert result_str == "none"

            print(f"[{network_label}] {labels[i]} split into {len(res.stars)} polys")

            for star in res.stars:
                # add constaint that output < 0 (low risk)
                assert star.a_mat.shape[0] == 1, "single output should mean single row"
                row = star.a_mat[0]
                bias = star.bias[0]

                star.lpi.add_dense_row(row, -bias)
                #star.lpi.add_dense_row(-2*row, -bias - 1)

                if star.lpi.is_feasible():
                    lpi_polys[i].append(star.lpi)



        print(f"[{network_label}] lp_polys size: {tuple(len(poly) for poly in lpi_polys)}") 
        for label_0, polys_0, prob_0 in zip(labels, lpi_polys, probs):
            total_probability = 0
            print(f"[Calculating total probability]")
            for lpi in tqdm.tqdm(polys_0):
                total_probability += integrate(lpi, prob_0)

            print("total probability:", total_probability)
            for label_1, polys_1, prob_1 in zip(labels, lpi_polys, probs):
                if label_0 == label_1:
                    continue

                pref_prob = 0
                print(f"[Calculating Preference Probability]")
                for lpi in tqdm.tqdm(polys_1):
                    pref_prob += integrate(lpi, prob_1)
                print("preference probability:", pref_prob)

                adv_prob = 0
                print(f"[Calculating Advantage Probability]")
                for lpi_0, lpi_1 in tqdm.tqdm(tuple(product(polys_0, polys_1))):
                    intersection_lpi = compute_intersection_lpi(lpi_0, lpi_1)

                    if intersection_lpi.is_feasible():
                        adv_prob += integrate(intersection_lpi, prob_0)

                print("advantage probability:", adv_prob)
                results_dict[network_label]['Advantage'][f"{label_0},{label_1}"] = total_probability - adv_prob
                results_dict[network_label]['Preference'][f"{label_0},{label_1}"] = total_probability - pref_prob

                print(f"[{network_label}] {label_0} advantage over {label_1}: {total_probability - adv_prob}")
                print(f"[{network_label}] {label_0} preference over {label_1}: {total_probability - pref_prob}")
        results_dict[network_label]['Symmetric Difference'] = sum(results_dict[network_label]['Advantage'].values())
        results_dict[network_label]['AUC'] = auc
        with open(sys.argv[2], 'w') as h:
            json.dump(results_dict, h)



if __name__ == "__main__":
    main()
