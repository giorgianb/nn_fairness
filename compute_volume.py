"""
Code for computing volume from linear constraints using quad and LP
"""

from scipy.integrate import quad

import numpy as np

import swiglpk as glpk
import glpk_util
from polytope import Polytope, extreme, is_fulldim
from scipy.spatial import ConvexHull
from icecream import ic
import math
from scipy.special import factorial

import quickhull
import cdd

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

def dd_extreme(poly):
    try:
        A, b = poly.A, poly.b
        b_2d = b.reshape((b.shape[0], 1))
        linsys = cdd.Matrix(np.hstack([b_2d, -A]), number_type='fraction')
        linsys.rep_type = cdd.RepType.INEQUALITY
        P = cdd.Polyhedron(linsys)
        generators = P.get_generators()
        vertices = np.vectorize(float)(np.array(generators)[:,1:])
        active = np.argsort(np.abs(b[:, None] - (A @ vertices.T)).T, axis=-1)[:, :A.shape[1]]
    except Exception as e:
        print(f"Caught exception: {e}")
        print("Defaulting to qhull_extreme.")
        return qhull_extreme(poly)
    return vertices, active

def qhull_extreme(poly):
    try:
        vertices, active = quickhull.extreme(poly)
    except Exception:
        vertices = None
        active = None

    return vertices, active

def qhull_integrate_polytope(poly, extreme):
    """use scipy.quad to integrate a function over a polytope domain

    params are the leq constraints Ax <= b

    if func_to_integrate is None, assumes function is 1 everywhere (returns volume)
    """
    if not poly.minrep:
        print("compute_volme.qhull_integrate_polytope: Polytope must be in minimal representation")
    vertices, active = extreme(poly)
    if vertices is None:
        print("WARNING: Vertices is none!")
        return 0, None

    try:
        hull = ConvexHull(vertices)
        return hull.volume, vertices
    except Exception:
        return 0, None

def lawrence_integrate_polytope_slow(poly, extreme, coeffs=None, P=0):
    # Kept as a reference implementation
    #from quickhull import extreme
    #N = A.shape[1]
    #M = A.shape[0]
    # This ensures all variables are > 0
    #Ac = np.concatenate((A, -np.identity(N)), axis=0)
    #bc = np.concatenate((b, np.zeros(N)))
    #poly = Polytope(Ac, bc)
    if not poly.minrep:
        print("compute_volme.qhull_integrate_polytope: Polytope must be in minimal representation")

    vertices, active_constraints = extreme(poly)


    A = poly.A
    b = poly.b
    N = A.shape[1]
    M = A.shape[0]
    if coeffs is None:
        coeffs = np.ones(N)
    
    slack = np.identity(M)
    last_row = np.zeros((1, A.shape[1] + M + 1))
    tableu = np.concatenate((A[:M], slack, b.reshape(b.shape[0], 1)[:M]), axis=-1)
    tableu = np.concatenate((tableu, last_row), axis=0)
    tableu[-1, :N] = -coeffs
    
    volume = 0
    for p, active_indices in zip(vertices, active_constraints):
        #active = np.isclose((A @ p)[:M], b[:M])
        active = np.zeros(M, dtype=bool)
        active[active_indices] = 1

        # Find the basic and non-basic variables
        basic_indices = np.arange(M)
        basic_indices[np.logical_not(active)] += N # These are the basic slack variables
        basic_indices[active] = np.arange(N)
        basic_indices = np.sort(basic_indices)
        possible_indices = np.arange(M + N)
        unused = np.ones(M + N).astype(np.bool_)
        unused[basic_indices] = False

        non_basic_indices = possible_indices[unused]

        δ_v = abs(np.linalg.det(tableu[:M, basic_indices]))

        # So now it's for the constraints that are active that we need to figure out
        # Which is a NB variable
        c = tableu[-1, :M+N]
        c_N = c[non_basic_indices]
        c_B = c[basic_indices]
        # Matrix for basic  variables

        A_B = tableu[:M, basic_indices]

        # Matrix for non-basic variables
        A_N = tableu[:M, non_basic_indices]

        # Reduced cost
        cost = c_N - c_B @ np.linalg.inv(A_B) @ A_N
        f_v = coeffs @ p
        volume += f_v**(N + P)/np.prod(cost) * 1/δ_v
    volume *= math.factorial(P)/math.factorial(N + P)
    return volume, vertices

def lawrence_integrate_polytope(poly, extreme, coeffs=None, P=0):
    if not poly.minrep:
        print("compute_volme.qhull_integrate_polytope: Polytope must be in minimal representation")

    vertices, active_constraints = extreme(poly)


    A = poly.A
    b = poly.b
    N = A.shape[1]
    M = A.shape[0]
    n_vertices = vertices.shape[0]

    if coeffs is None:
        coeffs = np.ones((1, N))
        P = np.array([0])

    
    slack = np.identity(M)
    last_row = np.zeros((coeffs.shape[0], A.shape[1] + M + 1))
    tableu = np.concatenate((A[:M], slack, b.reshape(b.shape[0], 1)[:M]), axis=-1)
    last_row[:, :N] = -coeffs
    
    active = np.zeros((n_vertices, M), dtype=bool)
    np.put_along_axis(active, active_constraints, np.ones((active_constraints.shape)), axis=-1)
    basic_indices = np.arange(M)[None, :].repeat(n_vertices, 0)
    basic_indices[~active] += N
    basic_indices[active] = np.arange(N)[None, :].repeat(n_vertices, 0).flatten()
    basic_indices = np.sort(basic_indices, axis=-1)
    possible_indices = np.arange(M + N)[None, :].repeat(n_vertices, 0)
    unused = np.ones((n_vertices, M + N)).astype(bool)
    np.put_along_axis(unused, basic_indices, np.zeros(basic_indices.shape), -1)

    non_basic_indices = possible_indices[unused].reshape(n_vertices, -1)
    δ_v = np.abs(np.linalg.det(tableu[:M, basic_indices].transpose(1, 0, 2)))

    c = last_row[:, :M+N]
    c_N = c[:, non_basic_indices].transpose(1, 0, 2)
    c_B = c[:, basic_indices].transpose(1, 0, 2)

    A_B = tableu[:M, basic_indices].transpose(1, 0, 2)

    # Matrix for non-basic variables
    A_N = tableu[:M, non_basic_indices].transpose(1, 0, 2)

    # Reduced cost
    cost = c_N - (c_B @ np.linalg.inv(A_B) @ A_N)
    # f_v will need to change
    f_v = np.einsum('...j,ij->...i', coeffs, vertices)
    volume = factorial(P) * np.sum(f_v**(N + P)[:, None]/np.prod(cost, axis=-1).T * 1/δ_v, axis=-1) / factorial(N + P)
    return volume, vertices
