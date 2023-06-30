from compute_volume import qhull_integrate_polytope, lawrence_integrate_polytope
from compute_volume import qhull_extreme, dd_extreme
import compute_volume
from nnenum.lpinstance import LpInstance
import swiglpk as glpk
import numpy as np
from icecream import ic
from polytope import reduce, Polytope

volume = qhull_integrate_polytope
extreme = qhull_extreme


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

        poly = reduce(Polytope(A, b))
        poly.minrep = True
        prob += volume(poly, extreme)*p

    return prob
