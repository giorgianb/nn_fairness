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
from compute_volume import quad_integrate_glpk_lp


from icecream import ic
import os.path
import pickle
import tqdm

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

    def __init__(self):
        cache_path = 'NN-verification/cache'
        random_seed = 0
        cache_file_path = os.path.join(cache_path, f'np-adult-data-rs={random_seed}.pkl')
        with open(cache_file_path, 'rb') as f:
            data_dict = pickle.load(f)

        X_train = data_dict["X_train"]

        self.ages  = make_distribution(X_train[:, 0])
        self.edu_number  = make_distribution(X_train[:, 1])
        self.hours_per_week  = make_distribution(X_train[:, 2])

        self.funcs = [make_linear_interpolation_func(d) for d in [self.ages, self.edu_number, self.hours_per_week]]

        self.volumes = []

        for func, data in zip(self.funcs, [self.ages, self.edu_number, self.hours_per_week]):
            v = quad(func, data[0][0], data[-1][0], limit=500)[0]
            self.volumes.append(v)

    def sample(self, age, edu_number):
        """get probability density at a point"""

        p = (self.funcs[0](age) / self.volumes[0])  \
            * (self.funcs[1](edu_number) / self.volumes[1])

        return p



#    def sample(self, age, edu_number, hours_per_week):
#        """get probability density at a point"""
#
#        p = (self.funcs[0](age) / self.volumes[0])  \
#            * (self.funcs[1](edu_number) / self.volumes[1]) \
#            * (self.funcs[2](hours_per_week) / self.volumes[2])
#
#        return p

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


    networks = [("NN-verification/results/adult-model_config-small-max_epoch=10-train_bs=32-random_seed=0-is_random_weight-False-race_permute=False-sex_permute=False-both_sex_race_permute=False/model.onnx", "Seed 0")]#,
#                ("seed1.onnx", "Seed 1"),
#                ("seed2.onnx", "Seed 2")]

    # ideas:
    # Initial set defined as star: (triangle), where unused input dimension is the pdf
    #
    # then, to integrate you just compute the area at the end
    #
    # symmetric difference = area1 + area2 - 2*area of intersection
    # area of intersection can be optimized using zonotope box bounds

    prob_density = ProbabilityDensityComputer()

    for onnx_filename, network_label in networks:
        network = load_onnx_network_optimized(onnx_filename)

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

        male_inits = [aam_box, wm_box]
        female_inits = [aaf_box, wf_box]
        inits_list = [male_inits] #[male_inits, female_inits]
        sex_labels = ['Male'] #['Male', 'Female']

        for inits, sex in zip(inits_list, sex_labels):
            lpi_polys = [[], []]
            labels = ['Low Risk African American', 'Low Risk White', 'Union'] 

            for i, init in enumerate(inits + [None]):
                total_probability = 0

                if i < 2:
                    init_box = np.array(init, dtype=np.float32)

                    res = enumerate_network(init_box, network)
                    result_str = res.result_str
                    assert result_str == "none"

                    print(f"{labels[i]} split into {len(res.stars)} polys")

                    for star in tqdm.tqdm(res.stars):
                        # add constaint that output < 0 (low risk)
                        assert star.a_mat.shape[0] == 1, "single output should mean single row"
                        row = star.a_mat[0]
                        bias = star.bias[0]

                        star.lpi.add_dense_row(row, -bias)

                        if star.lpi.is_feasible():
                            # get verts in input space
                            lpi_polys[i].append(star.lpi)

                            #prob_density.sample       
                            total_probability += quad_integrate_glpk_lp(star.lpi.lp, prob_density.sample)

                            #supp_pt_func = lambda vec: star.lpi.minimize(-vec)

                            #verts = kamenev.get_verts(2, supp_pt_func)

                else:
                    print(f"lp_polys size: {len(lpi_polys[0]), len(lpi_polys[1])}") 
                    # compute union of lp_polys
                    for lpi1 in lpi_polys[0]:
                        for lpi2 in lpi_polys[1]:
                            #intersection_poly = glpk_util.intersect(poly1, poly2)

                            intersection_lpi = compute_intersection_lpi(lpi1, lpi2)

                            if intersection_lpi.is_feasible():
                                total_probability += quad_integrate_glpk_lp(intersection_lpi.lp, prob_density.sample)



                print(f"{labels[i]} probability: {total_probability}")
                


if __name__ == "__main__":
    main()
