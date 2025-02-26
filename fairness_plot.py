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
from joblib import Parallel, delayed


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

def integrate(lpi, pdf, fixed_indices):
    prob = 0

    A_lpi = lpi.get_constraints_csr().toarray()
    b_lpi = lpi.get_rhs()
    lpi_copy = LpInstance(lpi)
   
    for region in pdf.regions:
        A = A_lpi.copy()
        b = b_lpi.copy()
        # check if it's feasible before computing volume
        col_index = 0
        A_col_index = 0
        for (lbound, ubound) in region:
            if lbound == ubound and type(lbound) != tuple:
                if col_index not in fixed_indices:
                    glpk.glp_set_col_bnds(lpi_copy.lp, A_col_index + 1, glpk.GLP_FX, lbound, lbound) # needs: import swiglpk as glpk
                    A_col_index += 1
                col_index += 1
            # Handle one-hot type
            elif type(lbound) == tuple:
                for val in lbound:
                    if col_index not in fixed_indices:
                        glpk.glp_set_col_bnds(lpi_copy.lp, A_col_index + 1, glpk.GLP_FX, val, val) # needs: import swiglpk as glpk
                        A_col_index += 1
                    col_index += 1
            else:
                if col_index not in fixed_indices:
                    glpk.glp_set_col_bnds(lpi_copy.lp, A_col_index + 1, glpk.GLP_DB, lbound, ubound) # needs: import swiglpk as glpk
                    A_col_index += 1
                col_index += 1

        feasible = lpi_copy.is_feasible()
        if not feasible:
            continue

        point = []
        to_eliminate_cols = []
        to_eliminate_vals = []
        to_keep_cols = []
        col_index = 0
        A_col_index = 0
        for i, (lbound, ubound) in enumerate(region):
            p = lbound if lbound == ubound else (lbound + ubound) / 2
            if lbound == ubound and type(p) != tuple:
                if col_index not in fixed_indices:
                    to_eliminate_cols.append(A_col_index)
                    to_eliminate_vals.append(lbound)
                    A_col_index += 1
                col_index += 1
                point.append(p)
            elif type(p) == tuple:
                for val in p:
                    if col_index not in fixed_indices:
                        to_eliminate_cols.append(A_col_index)
                        to_eliminate_vals.append(val)
                        A_col_index += 1
                    col_index += 1
                point.extend(p)
            else:
                row = np.zeros((1, A.shape[1]))
                row[0, A_col_index] = 1
                A = np.concatenate((A, row, -row), axis=0)
                b = np.append(b, (ubound, -lbound))
                if col_index not in fixed_indices:
                    to_keep_cols.append(A_col_index)
                    A_col_index += 1
                col_index += 1
                point.append(p)

        p = pdf.sample(*point)
        # For volumetric fairness
        #p = 1
        if p == 0:
            continue

        if len(to_eliminate_cols) > 0:
            to_eliminate_vals = np.array(to_eliminate_vals)
            b -= A[:, to_eliminate_cols] @ to_eliminate_vals
            A = A[:, to_keep_cols]

        prob += qhull_integrate_polytope(A, b)*p

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

def make_one_hot_distribution(data):
    dist = defaultdict(int)

    for x in data:
        hot_index = np.argmax(x == 1)
        dist[hot_index] += 1

    return dist


def make_discrete_func(dist):
    def func(x):
        return dist.get(x, 0)

    return func

def make_one_hot_func(dist):
    def func(x):
        hot_index = np.argmax(x == 1)
        return dist.get(hot_index, 0)

    return func

def one_hot(hot_index, length):
    h = [0]*length
    h[hot_index] = 1
    return tuple(h)


class ProbabilityDensityComputer:
    """computes probability of input at given point"""

    def __init__(self, X, discrete_indices, continuous_indices, one_hot_indices, fixed_indices, class_filter):
        # Assumption: One-Hot indices are contiguous in the array for a one-hot feature
        matches_given = np.apply_along_axis(class_filter, 1, X)
        X = X[matches_given]

        self.discrete_indices = tuple(discrete_indices)
        self.continuous_indices = tuple(continuous_indices)
        self.one_hot_indices = tuple(one_hot_indices)
        self.fixed_indices = tuple(fixed_indices)
        self.continuous = [make_continuous_distribution(X[:, i]) for i in continuous_indices]
        self.discrete = [make_discrete_distribution(X[:, i]) for i in discrete_indices]
        self.one_hot = [make_one_hot_distribution(X[:, index_group]) for index_group in one_hot_indices]

        self.continuous_funcs = [make_linear_interpolation_func(d) for d in self.continuous]
        self.discrete_funcs = [make_discrete_func(d) for d in self.discrete]
        self.one_hot_funcs = [make_one_hot_func(d) for d in self.one_hot]

        self.continuous_volumes = [quad(f, d[0][0], d[-1][0], limit=500)[0] for f, d in zip(self.continuous_funcs, self.continuous)]
        self.discrete_volumes = [sum(f(k) for k in d.keys()) for f, d in zip(self.discrete_funcs, self.discrete)]
        self.one_hot_volumes = [sum(f(k) for k in d.keys()) for f, d in zip(self.one_hot_funcs, self.one_hot)]
        self.continuous_bounds = [tuple(zip(d[:, 0], d[1:, 0])) for d in self.continuous]
        self.discrete_bounds = [[(k, k) for k in sorted(d.keys())] for d in self.discrete]
        self.one_hot_bounds = [[(one_hot(k, len(d.keys())), one_hot(k, len(d.keys()))) for k in sorted(d.keys())] for d in self.one_hot]
        regions = sorted(
                chain(
                    zip([[k] for k in continuous_indices], self.continuous_bounds),
                    zip([[k] for k in discrete_indices], self.discrete_bounds),
                    zip(one_hot_indices, self.one_hot_bounds)
                )
        )
        self._regions = tuple(map(lambda x: x[1], regions))


    def sample(self, *args):
        """get probability density at a point"""

        p = 1
        x = np.array(args)
        for func, volume, index in zip(self.continuous_funcs, self.continuous_volumes, self.continuous_indices):
            p *= func(x[index])/volume

        for func, volume, index in zip(self.discrete_funcs, self.discrete_volumes, self.discrete_indices):
            p *= func(x[index])/volume

        for func, volume, index_group in zip(self.one_hot_funcs, self.one_hot_volumes, self.one_hot_indices):
            p *= func(x[index_group])/volume

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
    results = Parallel(n_jobs=16)(delayed(run_on_model)(config, i) for i in range(n_models))
    #print(sum(results))
    #return 

    results_dict = {}
    for result in results:
        for k, v in result.items():
            results_dict[k] = v

    with open(sys.argv[2], 'w') as handle:
        print(f"Saving result to: {sys.argv[2]}")
        json.dump(results_dict, handle)


if __name__ == "__main__":
    main()
