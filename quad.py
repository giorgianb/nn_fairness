from compute_volume import qhull_integrate_polytope
from nnenum.lpinstance import LpInstance
import swiglpk as glpk
import numpy as np
from icecream import ic

def get_bounding_box(A, b):
    lpi = LpInstance()
    lpi.add_cols(list(range(A.shape[1])))
    for row, rhs in zip(A, b):
        lpi.add_dense_row(row, rhs, normalize=False)

    bb = []
    for i in range(A.shape[1]):
        v = np.zeros(A.shape[1])
        v[i] = 1
        try:
            min_bound = lpi.minimize(v)[i]
            max_bound = lpi.minimize(-v)[i]
        except RuntimeError:
            ic(A, b)
            raise
        bb.append((min_bound, max_bound))

    bb = np.array(bb)
    return bb

def bounding_box_volume(bb):
    return np.prod(np.abs(bb[:, 1] - bb[:, 0]))

def bounding_box(lpi, pdf, fixed_indices):
    prob = 0

    A_lpi = lpi.get_constraints_csr().toarray()
    b_lpi = lpi.get_rhs()
    lpi_copy = LpInstance(lpi)

   
    for region in pdf.regions:
        # check if it's feasible before computing volume
        col_index = 0
        A_col_index = 0
        for (lbound, ubound) in region:
            # Handle Discrete Type
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
                # We don't set any bounds for continuous-types in this method
                if col_index not in fixed_indices:
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

        # Indices to update with the computed bounding box center
        update_indices = []
        for i, (lbound, ubound) in enumerate(region):
            p = lbound if lbound == ubound else (lbound + ubound) / 2
            # Discrete Case
            if lbound == ubound and type(p) != tuple:
                if col_index not in fixed_indices:
                    to_eliminate_cols.append(A_col_index)
                    to_eliminate_vals.append(lbound)
                    A_col_index += 1
                col_index += 1
                point.append(p)
            # One-Hot Case
            elif type(p) == tuple:
                for val in p:
                    if col_index not in fixed_indices:
                        to_eliminate_cols.append(A_col_index)
                        to_eliminate_vals.append(val)
                        A_col_index += 1
                    col_index += 1
                point.extend(p)
            # Continuous Case
            else:
                v = np.zeros(A_lpi.shape[1])
                v[col_index] = 1

                min_bound = lpi.minimize(v)[col_index]
                max_bound = lpi.minimize(-v)[col_index]

                if col_index not in fixed_indices:
                    to_keep_cols.append(A_col_index)
                    A_col_index += 1
                col_index += 1

                point.append(None)
                update_indices.append(i)

        A = A_lpi.copy()
        b = b_lpi.copy()
        if len(to_eliminate_cols) > 0:
            to_eliminate_vals = np.array(to_eliminate_vals)
            b -= A[:, to_eliminate_cols] @ to_eliminate_vals
            A = A[:, to_keep_cols]

        bb = get_bounding_box(A, b)
        assert len(bb) == len(update_indices)
        for update_index, (lbound, ubound) in zip(update_indices, bb):
            point[update_index] = (lbound + ubound) / 2

        p = pdf.sample(*point)

        if p == 0:
            continue

        prob += bounding_box_volume(bb)*p

    return prob


def block_qhull(lpi, pdf, fixed_indices):
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

