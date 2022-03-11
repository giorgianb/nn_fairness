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
            prob += qhull_integrate_polytope(A, b)*pdf.sample(*point)

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

class ProbabilityDensityComputer:
    """computes probability of input at given point"""

    def __init__(self, given=lambda x: True):
        cache_path = 'NN-verification/cache'
        random_seed = 0
        cache_file_path = os.path.join(cache_path, f'np-adult-data-rs={random_seed}.pkl')
        with open(cache_file_path, 'rb') as f:
            data_dict = pickle.load(f)

        X_train = data_dict["X_train"]
        matches_given = np.apply_along_axis(given, 1, X_train)
        X_train = X_train[matches_given]

        self.ages  = make_distribution(X_train[:, 0])
        self.edu_number  = make_distribution(X_train[:, 1])
        self.hours_per_week  = make_distribution(X_train[:, 2])

        self.funcs = [make_linear_interpolation_func(d) for d in [self.ages, self.edu_number, self.hours_per_week]]

        self.volumes = []

        for func, data in zip(self.funcs, [self.ages, self.edu_number, self.hours_per_week]):
            v = quad(func, data[0][0], data[-1][0], limit=500)[0]
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
        #jhour_bounds = zip(self.hours_per_week[:, 0], self.hours_per_week[1:, 0])

        return product(age_bounds, edu_bounds)#, hour_bounds)




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

    # Model Input:  [age, edu_num, hours_per_week, is_male, race_white, race_black, race_asian_pac_islander, race_amer_indian_eskimo, race_other]
    # age             :       normalized to [0,1], original max: 90 , original min: 17
    # edu_num         :       normalized to [0,1], original max: 16 , original min: 1
    # hours_per_week  :       normalized to [0,1], original max: 99 , original min: 1
    # is_male         : 1.0 means male, 0.0 means female,
    # race-related:   : mutually exclusive indicator (with either 1.0 or 0.0)

    # output interpretation: is_greater_than_50K: Binary indictor

    networks = [
            ("NN-verification/results/adult-model_config-small-max_epoch=10-train_bs=32-random_seed=0-is_random_weight-False-race_permute=False-sex_permute=False-both_sex_race_permute=False/model.onnx", ("Small", "None")),
            ("NN-verification/results/adult-model_config-small-max_epoch=10-train_bs=32-random_seed=0-is_random_weight-False-race_permute=True-sex_permute=False-both_sex_race_permute=False/model.onnx", ("Small", "Race Permute")),
            ("NN-verification/results/adult-model_config-small-max_epoch=10-train_bs=32-random_seed=0-is_random_weight-False-race_permute=False-sex_permute=True-both_sex_race_permute=False/model.onnx", ("Small", "Sex Permute")), 
            ("NN-verification/results/adult-model_config-small-max_epoch=10-train_bs=32-random_seed=0-is_random_weight-False-race_permute=False-sex_permute=False-both_sex_race_permute=True/model.onnx", ("Small", "Both Permute")), 
            ("NN-verification/results/adult-model_config-small-max_epoch=10-train_bs=32-random_seed=0-is_random_weight-True-race_permute=False-sex_permute=False-both_sex_race_permute=False/model.onnx", ("Small", "Random Weight")),
            ("NN-verification/results/adult-model_config-medium-max_epoch=10-train_bs=32-random_seed=0-is_random_weight-False-race_permute=False-sex_permute=False-both_sex_race_permute=False/model.onnx",("Medium", "None")),
            ("NN-verification/results/adult-model_config-medium-max_epoch=10-train_bs=32-random_seed=0-is_random_weight-False-race_permute=True-sex_permute=False-both_sex_race_permute=False/model.onnx",("Medium", "Race Permute")),
            ("NN-verification/results/adult-model_config-medium-max_epoch=10-train_bs=32-random_seed=0-is_random_weight-False-race_permute=False-sex_permute=True-both_sex_race_permute=False/model.onnx",("Medium", "Sex Permute")),
            ("NN-verification/results/adult-model_config-medium-max_epoch=10-train_bs=32-random_seed=0-is_random_weight-False-race_permute=False-sex_permute=False-both_sex_race_permute=True/model.onnx",("Medium", "Both Permute")),
            ("NN-verification/results/adult-model_config-medium-max_epoch=10-train_bs=32-random_seed=0-is_random_weight-True-race_permute=False-sex_permute=False-both_sex_race_permute=False/model.onnx",("Medium", "Random Weight")),
    ]

    # ideas:
    # Initial set defined as star: (triangle), where unused input dimension is the pdf
    #
    # then, to integrate you just compute the area at the end
    #
    # symmetric difference = area1 + area2 - 2*area of intersection
    # area of intersection can be optimized using zonotope box bounds

    def is_aam(x):
        return x[3] == 1 and x[5] == 1

    def is_wm(x):
        return x[3] == 1 and x[4] == 1

    def is_apim(x):
        return x[3] == 1 and x[5] == 1

    def is_aiem(x):
        return x[3] == 1 and x[6] == 1

    aam_prob = ProbabilityDensityComputer(is_aam)
    wm_prob = ProbabilityDensityComputer(is_wm)
    apim_prob = ProbabilityDensityComputer(is_apim)
    aiem_prob = ProbabilityDensityComputer(is_aiem)
    aam_box = [
            [0.0, 1.0], # age
            [0.0, 1.0], # edu_num
            [1.0, 1.0], # hours_per_week
            [1.0, 1.0], # is_male
            [0.0, 0.0], # race_white
            [1.0, 1.0], # race_black
            [0.0, 0.0], # race_asian_pac_islander
            [0.0, 0.0], # race_american_indian_eskimo
            [0.0, 0.0], # race_other
    ]

    wm_box = [
            [0.0, 1.0], # age
            [0.0, 1.0], # edu_num
            [1.0, 1.0], # hours_per_week
            [1.0, 1.0], # is_male
            [1.0, 1.0], # race_white
            [0.0, 0.0], # race_black
            [0.0, 0.0], # race_asian_pac_islander
            [0.0, 0.0], # race_american_indian_eskimo
            [0.0, 0.0], # race_other
    ]

    apim_box = [
            [0.0, 1.0], # age
            [0.0, 1.0], # edu_num
            [1.0, 1.0], # hours_per_week
            [1.0, 1.0], # is_male
            [0.0, 0.0], # race_white
            [0.0, 0.0], # race_black
            [1.0, 1.0], # race_asian_pac_islander
            [0.0, 0.0], # race_american_indian_eskimo
            [0.0, 0.0], # race_other
    ]

    aiem_box = [
            [0.0, 1.0], # age
            [0.0, 1.0], # edu_num
            [1.0, 1.0], # hours_per_week
            [1.0, 1.0], # is_male
            [0.0, 0.0], # race_white
            [0.0, 0.0], # race_black
            [0.0, 0.0], # race_asian_pac_islander
            [1.0, 1.0], # race_american_indian_eskimo
            [0.0, 0.0], # race_other
    ]

    aaf_box = [
            [0.0, 1.0], # age
            [0.0, 1.0], # edu_num
            [0.0, 1.0], # hours_per_week
            [0.0, 0.0], # is_male
            [0.0, 0.0], # race_white
            [1.0, 1.0], # race_black
            [0.0, 0.0], # race_asian_pac_islander
            [0.0, 0.0], # race_american_indian_eskimo
            [0.0, 0.0], # race_other
    ]

    wf_box = [
            [0.0, 1.0], # age
            [0.0, 1.0], # edu_num
            [0.0, 1.0], # hours_per_week
            [0.0, 0.0], # is_male
            [1.0, 1.0], # race_white
            [0.0, 0.0], # race_black
            [0.0, 0.0], # race_asian_pac_islander
            [0.0, 0.0], # race_american_indian_eskimo
            [0.0, 0.0], # race_other
    ]


    rows = [('size', 'action', 'metric', 'r1', 'r2', 'value')]
    for onnx_filename, network_label in networks:
        network = load_onnx_network_optimized(onnx_filename)

        male_inits = [aam_box, wm_box]#, apim_box, aiem_box]
        female_inits = [aaf_box, wf_box]
        inits_list = [male_inits] #[male_inits, female_inits]
        male_probs = [aam_prob, wm_prob]#, apim_prob, aiem_prob]
        probs_list = [male_probs]
        sex_labels = ['Male'] #['Male', 'Female']

        # Here, we calculate the intersection
        for inits, probs, sex in zip(inits_list, probs_list, sex_labels):
            lpi_polys = []
            labels = ['Black', 'White', 'Asiatic', 'Native American'] 
            total_probabilities = []

            for i, (init, prob) in enumerate(zip(inits, probs)):
                lpi_polys.append([])

                init_box = np.array(init, dtype=np.float32)

                res = enumerate_network(init_box, network)
                result_str = res.result_str
                assert result_str == "none"

                print(f"[{network_label}] {labels[i]} split into {len(res.stars)} polys")

                for star in tqdm.tqdm(res.stars):
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
                for lpi in polys_0:
                    total_probability += integrate(lpi, prob_0)

                for label_1, polys_1, prob_1 in zip(labels, lpi_polys, probs):
                    if label_0 == label_1:
                        continue

                    pref_prob = 0
                    for lpi in polys_1:
                        pref_prob += integrate(lpi, prob_1)

                    adv_prob = 0
                    for lpi_0, lpi_1 in tqdm.tqdm(tuple(product(polys_0, polys_1))):
                        intersection_lpi = compute_intersection_lpi(lpi_0, lpi_1)

                        if intersection_lpi.is_feasible():
                            adv_prob += integrate(intersection_lpi, prob_0)

                    rows.append((network_label[0], network_label[1], 'Advantage', label_0, label_1, total_probability - adv_prob))
                    rows.append((network_label[0], network_label[1], 'Preference', label_0, label_1, total_probability - pref_prob))

                    print(f"[{network_label}] {label_0} advantage over {label_1}: {total_probability - adv_prob}")
                    print(f"[{network_label}] {label_0} preference over {label_1}: {total_probability - pref_prob}")

    with open('results.csv', 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


if __name__ == "__main__":
    main()
