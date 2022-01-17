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

class ProbabilityDensityComputer:
    """computes probability of input at given point"""

    def __init__(self):
        self.ages = [(0, 50), (0.2, 100), (1.0, 0)]
        self.priors = [(0, 20), (0.2, 5), (1.0, 0)]

        self.funcs = [make_linear_interpolation_func(d) for d in [self.ages, self.priors]]

        self.volumes = []

        for func, data in zip(self.funcs, [self.ages, self.priors]):
            v = quad(func, data[0][0], data[-1][0])[0]
            self.volumes.append(v)

    def sample(self, age, priors):
        """get probability density at a point"""

        return (self.funcs[0](age) / self.volumes[0]) * (self.funcs[1](priors) / self.volumes[1])

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

    #Model Input: [ age, prior, race, sex]
    #- age and prior is normailzed to be 0~1. 
    #    - you can see the joint distribution in this plot: NN-verfication/results/fig-feature-2d-distribution-rs=0.pdf )
    #- race and sex is in binary. 
    #    - race=1 -> African-American 
    #    - race=0 -> White
    #    - sex=1 ->  Male
    #    - sex=0 ->  Female

    # output interpretation: <= 0 mean jail, >= 0 means bail

    # first pass: compute area of aa-men + aa-women
    # compare with area of w-men + w-women

    networks = [("seed0.onnx", "Seed 0")]#,
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

        aam_box = [[0.0, 1.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
        wm_box = [[0.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 1.0]]

        aaf_box = [[0.0, 1.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]
        wf_box = [[0.0, 1.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]

        male_inits = [aam_box, wm_box]
        female_inits = [aaf_box, wf_box]
        inits_list = [male_inits] #[male_inits, female_inits]
        sex_labels = ['Male'] #['Male', 'Female']

        for inits, sex in zip(inits_list, sex_labels):
            fig, ax_list = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
            fig.suptitle(f'Computed Exact Sets ({sex}, {network_label})', fontsize=20)
        
            colors = ['r', 'b', 'g']
            labels = ['Low Risk African American', 'Low Risk White', 'Union']

            lpi_polys = [[], []]

            for i, init in enumerate(inits + [None]):
                ax = ax_list[i]
                ax.set_title(labels[i], fontsize=14)
                total_probability = 0

                if i < 2:
                    init_box = np.array(init, dtype=np.float32)

                    res = enumerate_network(init_box, network)
                    result_str = res.result_str
                    assert result_str == "none"

                    print(f"{labels[i]} split into {len(res.stars)} polys")

                    for star in res.stars:
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

                            supp_pt_func = lambda vec: star.lpi.minimize(-vec)

                            verts = kamenev.get_verts(2, supp_pt_func)

                            ax.fill(*zip(*verts), alpha=0.2, ec=colors[i], fc=colors[i])
                            #ax.plot(*zip(*verts), alpha=0.5, color=colors[i])
                else:
                    print(f"lp_polys size: {len(lpi_polys[0]), len(lpi_polys[1])}") 
                    # compute union of lp_polys
                    for lpi1 in lpi_polys[0]:
                        for lpi2 in lpi_polys[1]:
                            #intersection_poly = glpk_util.intersect(poly1, poly2)

                            intersection_lpi = compute_intersection_lpi(lpi1, lpi2)

                            if intersection_lpi.is_feasible():
                                total_probability += quad_integrate_glpk_lp(intersection_lpi.lp, prob_density.sample)

                                supp_pt_func = lambda vec: intersection_lpi.minimize(-vec)

                                verts = kamenev.get_verts(2, supp_pt_func)

                                ax.fill(*zip(*verts), alpha=0.5, color=colors[i], fc=colors[i])


                print(f"{labels[i]} probability: {total_probability}")
                ax.text(0.02, 0.98, f"Computed Probability: {round(total_probability, 6)}", va='top')
                
                ax.set_xlabel('Age')
                ax.set_ylabel('Priors')
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])

            #ax.legend()
            # x axis = age
            # y axis = prior
            plot_filename = f"{sex}_{network_label.replace(' ', '_')}.png"
            fig.savefig(plot_filename, bbox_inches='tight')
            print(f"Wrote {plot_filename}")

if __name__ == "__main__":
    main()
