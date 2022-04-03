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

from collections import defaultdict

import swiglpk as glpk

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

def make_distribution(data):
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

    def __init__(self, given=lambda x: True):
        cache_path = 'NN-verification/cache'
        random_seed = 0
        cache_file_path = os.path.join(cache_path, f'np-adult-data-v2-rs={random_seed}.pkl')
        with open(cache_file_path, 'rb') as f:
            data_dict = pickle.load(f)


        X_train = data_dict["X_train"]
        matches_given = np.apply_along_axis(given, 1, X_train)
        X_train = X_train[matches_given]

        #[age_feat 0, edu_feat 1, hours_per_week_feat 2, sex_feat 3, race_white_black_feat 4, country_is_native_feat 5, occupation_managerial_feat 6, occupation_is_gov_feat 7]
        
        self.ages = make_distribution(X_train[:, 0])
        self.edu_number = make_distribution(X_train[:, 1])
        self.hours_per_week = make_distribution(X_train[:, 2])
        self.is_native = make_discrete_distribution(X_train[:, 5])
        self.is_manager = make_discrete_distribution(X_train[:, 6])
        self.is_gov = make_discrete_distribution(X_train[:, 7])

        # Continuous Functions
        self.cont_funcs = [make_linear_interpolation_func(d) for d in [self.ages, self.edu_number, self.hours_per_week]]
        # Discrete functions
        self.disc_funcs = [make_discrete_func(d) for d in [self.is_native, self.is_manager, self.is_gov]]
        self.funcs = self.cont_funcs + self.disc_funcs

        self.volumes = []

        for func, data in zip(self.funcs, [self.ages, self.edu_number, self.hours_per_week]):
            v = quad(func, data[0][0], data[-1][0], limit=500)[0]
            self.volumes.append(v)

        for func, data in zip(self.disc_funcs, [self.is_native, self.is_manager, self.is_gov]):
            v = 0
            for k in data.keys():
                v += func(k)
            self.volumes.append(v)

    def sample(self, *args):
        """get probability density at a point"""

        p = 1
        for func, volume, x in zip(self.funcs, self.volumes, args):
            p *= func(x)/volume

        return p

    @property
    def regions(self):
        age_bounds = zip(self.ages[:, 0], self.ages[1:, 0])
        edu_bounds = zip(self.edu_number[:, 0], self.edu_number[1:, 0])
        hour_bounds = zip(self.hours_per_week[:, 0], self.hours_per_week[1:, 0])
        native_bounds = zip(sorted(self.is_native.keys()), sorted(self.is_native.keys()))
        manager_bounds = zip(sorted(self.is_manager.keys()), sorted(self.is_manager.keys()))
        gov_bounds = zip(sorted(self.is_gov.keys()), sorted(self.is_gov.keys()))


        return product(age_bounds, edu_bounds, hour_bounds, native_bounds, manager_bounds, gov_bounds)


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


    networks = [
            ("NN-verification/results/adult-v2-model_config-small-max_epoch=10-train_bs=32-random_seed=0-is_random_weight-False-race_permute=False-sex_permute=False-both_sex_race_permute=False/model.onnx", ("S", "N")),
            ("NN-verification/results/adult-v2-model_config-small-max_epoch=10-train_bs=32-random_seed=0-is_random_weight-False-race_permute=True-sex_permute=False-both_sex_race_permute=False/model.onnx", ("S", "RP")),
            ("NN-verification/results/adult-v2-model_config-small-max_epoch=10-train_bs=32-random_seed=0-is_random_weight-False-race_permute=False-sex_permute=True-both_sex_race_permute=False/model.onnx", ("S", "SP")), 
            ("NN-verification/results/adult-v2-model_config-small-max_epoch=10-train_bs=32-random_seed=0-is_random_weight-False-race_permute=False-sex_permute=False-both_sex_race_permute=True/model.onnx", ("S", "BP")), 
            ("NN-verification/results/adult-v2-model_config-small-max_epoch=10-train_bs=32-random_seed=0-is_random_weight-True-race_permute=False-sex_permute=False-both_sex_race_permute=False/model.onnx", ("S", "RW")),
            ("NN-verification/results/adult-v2-model_config-medium-max_epoch=10-train_bs=32-random_seed=0-is_random_weight-False-race_permute=False-sex_permute=False-both_sex_race_permute=False/model.onnx",("M", "N")),
            ("NN-verification/results/adult-v2-model_config-medium-max_epoch=10-train_bs=32-random_seed=0-is_random_weight-False-race_permute=True-sex_permute=False-both_sex_race_permute=False/model.onnx",("M", "RP")),
            ("NN-verification/results/adult-v2-model_config-medium-max_epoch=10-train_bs=32-random_seed=0-is_random_weight-False-race_permute=False-sex_permute=True-both_sex_race_permute=False/model.onnx",("M", "SP")),
            ("NN-verification/results/adult-v2-model_config-medium-max_epoch=10-train_bs=32-random_seed=0-is_random_weight-False-race_permute=False-sex_permute=False-both_sex_race_permute=True/model.onnx",("M", "BP")),
            ("NN-verification/results/adult-v2-model_config-medium-max_epoch=10-train_bs=32-random_seed=0-is_random_weight-True-race_permute=False-sex_permute=False-both_sex_race_permute=False/model.onnx",("M", "RW")),
    ]

    # ideas:
    # Initial set defined as star: (triangle), where unused input dimension is the pdf
    #
    # then, to integrate you just compute the area at the end
    #
    # symmetric difference = area1 + area2 - 2*area of intersection
    # area of intersection can be optimized using zonotope box bounds

    def is_aam(x):
        return x[3] == 1 and x[4] == 1

    def is_wm(x):
        return x[3] == 1 and x[4] == 0


    aam_prob = ProbabilityDensityComputer(is_aam)
    wm_prob = ProbabilityDensityComputer(is_wm)
# x = [age_feat, edu_feat, hours_per_week_feat, sex_feat, race_white_black_feat, country_is_native_feat, occupation_managerial_feat, occupation_is_gov_feat]
# y = [ is_greater_than_50K ]
# 
# age             :       normalized to [0,1], original max: 90 , original min: 17
# edu_num         :       normalized to [0,1], original max: 16 , original min: 1
# hours_per_week  :       normalized to [0,1], original max: 99 , original min: 1
# is_male         : 1.0 means male, 0.0 means female,
# race-related:   : 1.0 means black, 0.0 means white,
# native_immigrant: 1.0 means American, 0.0 means not,
# managerial_feat : 1.0 means management job, 0.0 means not,
# gov_feat        : 1.0 means government job,
    # output interpretation: is_greater_than_50K: Binary indictor

    aam_box = [
            [0.0, 1.0], # age
            [0.0, 1.0], # edu_num
            [0.0, 1.0], # hours_per_week
            [1.0, 1.0], # is_male
            [1.0, 1.0], # is_black
            [0.0, 1.0], # native immigrant
            [0.0, 1.0], # manager
            [0.0, 1.0], # govt feat
    ]

    wm_box = [
            [0.0, 1.0], # age
            [0.0, 1.0], # edu_num
            [0.0, 1.0], # hours_per_week
            [1.0, 1.0], # is_male
            [0.0, 0.0], # is_black
            [0.0, 1.0], # native immigrant
            [0.0, 1.0], # manager
            [0.0, 1.0], # govt feat
    ]


    # results_dict['model_size']['metric']['fairness_action']
    results_dict = defaultdict(lambda: defaultdict(dict))
    for onnx_filename, network_label in networks:
        network = load_onnx_network_optimized(onnx_filename)

        male_inits = [aam_box, wm_box]#, apim_box, aiem_box]
        male_probs = [aam_prob, wm_prob]#, apim_prob, aiem_prob]
        inits_list = [male_inits] #[male_inits, female_inits]
        probs_list = [male_probs]
        sex_labels = ['Male'] #['Male', 'Female']

        # Here, we calculate the intersection
        for inits, probs, sex in zip(inits_list, probs_list, sex_labels):
            lpi_polys = []
            labels = ['B', 'W', 'A', 'NA American'] 
            total_probabilities = []

            for i, (init, prob) in enumerate(zip(inits, probs)):
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
                    results_dict[network_label[0]][f'{label_0}A{label_1}'][network_label[1]] = total_probability - adv_prob
                    results_dict[network_label[0]][f'{label_0}P{label_1}'][network_label[1]] = total_probability - pref_prob

                    print(f"[{network_label}] {label_0} advantage over {label_1}: {total_probability - adv_prob}")
                    print(f"[{network_label}] {label_0} preference over {label_1}: {total_probability - pref_prob}")

    rows = []
    for size, size_results in results_dict.items():
        for metric, metric_results in size_results.items():
            metric_results = copy.deepcopy(metric_results)
            metric_results['metric'] = metric
            metric_results['size'] = size
            rows.append(metric_results)

    fieldnames = ('size', 'metric', 'N', 'RP', 'SP', 'BP', 'RW')
    with open('results_full_cdd.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
