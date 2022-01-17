"""
fairness analyzer exploration using nnenum

this one computes volumes

Nov 2021, Stanley Bak
"""

import sys

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull

import swiglpk as glpk

from polytope import Polytope, extreme, volume

from nnenum.enumerate import enumerate_network
from nnenum.settings import Settings
from nnenum.result import Result
from nnenum.onnx_network import load_onnx_network_optimized, load_onnx_network
from nnenum import kamenev

def set_settings():
    """exact analysis settings"""

    Settings.TIMING_STATS = False # True
    Settings.PRINT_OUTPUT = False
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

    matplotlib.use('TkAgg') # set backend
    plt.style.use(['bmh', 'bak_matplotlib.mlpstyle'])

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

    #networks = [("seed0.onnx", "Seed 0"),
    #            ("seed1.onnx", "Seed 1"),
    #            ("seed2.onnx", "Seed 2")]
    networks = [("seed0.onnx", "Seed 0")]

    # ideas:
    # Initial set defined as star: (triangle), where unused input dimension is the pdf
    #
    # then, to integrate you just compute the area at the end
    #
    # symmetric difference = area1 + area2 - 2*area of intersection
    # area of intersection can be optimized using zonotope box bounds


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
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_title(f'Computed Exact Set ({network_label})')
        
            colors = ['b', 'lime']
            labels = [f'Low Risk African American ({sex})', f'Low Risk White ({sex})']

            volumes = [0.0, 0.0]

            for i, init in enumerate(inits):
                init_box = np.array(init, dtype=np.float32)

                res = enumerate_network(init_box, network)
                result_str = res.result_str

                print(f"{labels[i]} split into {len(res.stars)} polys")
                first = True

                for star in res.stars:
                    # add constaint that output < 0 (low risk)
                    assert star.a_mat.shape[0] == 1, "single output should mean single row"
                    row = star.a_mat[0]
                    bias = star.bias[0]

                    star.lpi.add_dense_row(row, -bias)

                    if star.lpi.is_feasible():
                        vol = lpi_volume(star.lpi)
                        volumes[i] += vol
                        
                        # get verts in input space
                        supp_pt_func = lambda vec: star.lpi.minimize(-vec)

                        verts = kamenev.get_verts(2, supp_pt_func)

                        label = labels[i] if first else None
                        first = False
                        ax.fill(*zip(*verts), label=label, alpha=0.5, color=None, fc=colors[i])

            print(f"Volumes: {volumes}")
                        
            ax.set_xlabel('Age (normalized)')
            ax.set_ylabel('Prior (normalized)')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])

            ax.legend()
            # x axis = age
            # y axis = prior
            plot_filename = f"{sex}_{network_label.replace(' ', '_')}.png"
            fig.savefig(plot_filename)

def lpi_volume(lpi):
    """compute volume of a feasible lpinstance"""

    print(lpi)

    A = lpi.get_constraints_csr().toarray()
    b = lpi.get_rhs()

    dbl_max = sys.float_info.max

    for index in range(lpi.get_num_cols()):
        lb = glpk.glp_get_col_lb(lpi.lp, index + 1)
        ub = glpk.glp_get_col_ub(lpi.lp, index + 1)

        print("TODO: add bounds constraints")
        
        if lb != -dbl_max:
            pass

        if ub != dbl_max:
            pass

    #A = np.array([[0, -1], [-1, 0], [1, 1]], dtype=float)
    #b = np.array([0, 0, 1], dtype=float)

    poly = Polytope(A, b)

    print(poly)

    vertices = extreme(poly)

    #print(vertices)

    hull = ConvexHull(vertices)

    print(f"volume with random algorithm: {volume(poly)}, qhull volume: {hull.volume}")

    print("debug exit")
    exit(1)

    return hull.volume

if __name__ == "__main__":
    main()
