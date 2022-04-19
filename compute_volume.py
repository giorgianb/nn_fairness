"""
Code for computing volume from linear constraints using quad and LP
"""

from scipy.integrate import quad

import numpy as np

import swiglpk as glpk
import glpk_util
from polytope import Polytope, extreme
from scipy.spatial import ConvexHull
from icecream import ic
#import cdd

def rand_integrate_polytope(A, b, func_to_integrate=None, samples=100000):
    """use random sampling to integrate a function over a polytope domian

    params are the leq constraints Ax <= b

    if func_to_integrate is None, assumes function is 1 everywhere (returns volume)
    """

    assert isinstance(samples, int)

    if func_to_integrate is None:
        func_to_integrate = lambda *args: 1

    glpk_lp = glpk_util.from_constraints(A, b)

    total_func = 0
    box_bounds = glpk_util.lp_box_bounds(glpk_lp)

    deltas = np.array([ub - lb for ub, lb in box_bounds], dtype=float)
    lbs = np.array([lb for _, lb in box_bounds], dtype=float)

    box_volume = np.prod(deltas)
    dims = glpk_util.get_num_cols(glpk_lp)

    limits = [[np.inf, -np.inf] for _ in range(dims)]

    for _ in range(samples):
        rand_vals = np.random.rand(dims)

        # scale
        pt = np.multiply(rand_vals, deltas) + lbs

        if np.all(A @ pt <= b):
            #pts_in += 1
            total_func += func_to_integrate(*pt)

            for d, (lb, ub) in enumerate(limits):
                if pt[d] < lb:
                    limits[d][0] = pt[d]

                if pt[d] > ub:
                    limits[d][1] = pt[d]

    rv = total_func / samples * box_volume

    glpk.glp_delete_prob(glpk_lp)

    return rv

def quad_integrate_polytope(A, b, func_to_integrate=None):
    """use scipy.quad to integrate a function over a polytope domain

    params are the leq constraints Ax <= b

    if func_to_integrate is None, assumes function is 1 everywhere (returns volume)
    """

    lp = glpk_util.from_constraints(A, b)
    #print(glpk_util.to_str(lp))

    rv = quad_integrate_glpk_lp(lp, func_to_integrate)

    glpk.glp_delete_prob(lp)

    return rv

def get_A_b_vertices(A, b):
    assert len(b.shape)==1
    b_2d = b.reshape((b.shape[0], 1))
    linsys = cdd.Matrix(np.hstack([b_2d, -A]), number_type='float')
    linsys.rep_type = cdd.RepType.INEQUALITY
    P = cdd.Polyhedron(linsys)
    generators = P.get_generators()

    return np.array(generators)[:,1:]

def qhull_integrate_polytope(A, b, func_to_integrate=None):
    """use scipy.quad to integrate a function over a polytope domain

    params are the leq constraints Ax <= b

    if func_to_integrate is None, assumes function is 1 everywhere (returns volume)
    """
    try:
        vertices = get_A_b_vertices(A, b)
    except Exception:
        poly = Polytope(A, b)
        vertices = extreme(poly)

    if vertices is None:
        return 0

    hull = ConvexHull(vertices)

    return hull.volume

def quad_integrate_glpk_lp(glpk_lp_instance, func_to_integrate=None):
    """use scipy.quad to integrate a function over a polytope domain

    takes in polytope as a gplk lp instance

    if func_to_integrate is None, assumes function is 1 everywhere (returns volume)
    """

    if func_to_integrate is None:
        func_to_integrate = lambda *args: 1

    integrator = LpIntegrator(glpk_lp_instance)

    def eval_func(x, *args):
        """evaluate the function"""

        if len(args) + 1 < integrator.num_cols:
            dim_lb, dim_ub = integrator.get_bounds(*args, x)

            rv = quad(eval_func, dim_lb, dim_ub, args=(*args, x), limit=500)[0]
        else:
            # base case: last dimension
            rv = func_to_integrate(*args, x)

        return rv

    lb, ub = integrator.get_bounds()
    rv = quad(eval_func, lb, ub, limit=500)[0]

    return rv

class LpIntegrator:
    """an LP being integrated"""

    def __init__(self, lp):

        self.lp = lp
        self.num_cols = glpk_util.get_num_cols(lp)

    def get_bounds(self, *fixed_cols):
        """get bounds of the next column, given a list of fixed values for columns"""

        lp = glpk.glp_create_prob()
        glpk.glp_copy_prob(lp, self.lp, glpk.GLP_OFF)

        # fix the variables
        for i, val in enumerate(fixed_cols):
            glpk.glp_set_col_bnds(lp, i + 1, glpk.GLP_FX, float(val), float(val))
       
        direction_vec = [0] * self.num_cols

        cur_col = len(fixed_cols)
        assert cur_col < self.num_cols, f"get_bounds called with fixed_cols={fixed_cols}, but num_cols = {self.num_cols}"

        direction_vec[cur_col] = 1
        glpk_util.set_minimize_direction(lp, direction_vec)

        params = glpk_util.default_params()
        simplex_res = glpk.glp_simplex(lp, params)
        assert simplex_res == 0
        lb = glpk.glp_get_col_prim(lp, int(1 + cur_col))

        direction_vec[cur_col] = -1
        glpk_util.set_minimize_direction(lp, direction_vec)

        simplex_res = glpk.glp_simplex(lp, params)
        assert simplex_res == 0
        ub = glpk.glp_get_col_prim(lp, int(1 + cur_col))

        glpk.glp_delete_prob(lp)

        return lb, ub
