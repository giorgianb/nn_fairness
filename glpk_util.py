"""
Utilities for glpk interaction
"""

from typing import List, Tuple

import math
import sys

import numpy as np

from termcolor import colored
import swiglpk as glpk

def default_params():
    """get a static instance of the default params"""

    if not hasattr(default_params, "params"):
        params = glpk.glp_smcp()
        glpk.glp_init_smcp(params)

        #params.msg_lev = glpk.GLP_MSG_ERR
        params.msg_lev = glpk.GLP_MSG_ERR
        params.meth = glpk.GLP_PRIMAL # dual

        params.tm_lim = int(60 * 1000)
        params.out_dly = 2 * 1000 # start printing to terminal delay

        default_params.params = params

    return default_params.params
    
def lp_box_bounds(lp) -> List[Tuple[float, float]]:
    """get box bounds from constraints"""

    cols = get_num_cols(lp)
    direction = [0] * cols
    rv: List[Tuple[float, float]] = []
    params = default_params()

    for col in range(cols):
        direction[col] = 1
        set_minimize_direction(lp, direction)
        simplex_res = glpk.glp_simplex(lp, params)
        assert simplex_res == 0
        lb = glpk.glp_get_col_prim(lp, int(1 + col))

        direction[col] = -1
        set_minimize_direction(lp, direction)
        simplex_res = glpk.glp_simplex(lp, params)
        assert simplex_res == 0
        ub = glpk.glp_get_col_prim(lp, int(1 + col))

        direction[col] = 0

        assert lb <= ub
        rv.append((lb, ub))
                
    return rv

def from_constraints(A, b):
    """create GLPK lp object from Ax <= b constraints"""

    if not isinstance(A, np.ndarray):    
        A = np.array(A, dtype=float)

    if not isinstance(b, np.ndarray):
        b = np.array(b, dtype=float)

    rows, cols = A.shape

    assert len(b.shape) == 1, "expected 1-d vector for right hand side"
    assert b.shape[0] == rows, f"A matrix is {A.shape}, b vector needs {rows} elements, got {b.shape}"
    
    lp = glpk.glp_create_prob()

    # add columns
    glpk.glp_add_cols(lp, cols)

    for i in range(cols):
        glpk.glp_set_col_bnds(lp, i + 1, glpk.GLP_FR, 0, 0)  # free variable (-inf, inf)

    # add rows
    glpk.glp_add_rows(lp, rows)

    for i, rhs in enumerate(b):
        glpk.glp_set_row_bnds(lp, i + 1, glpk.GLP_UP, 0, rhs)  # '<=' constraint
            
    for i, vec in enumerate(A):
        data_vec = SwigArray.as_double_array(vec, vec.size)
        indices_vec = SwigArray.get_sequential_int_array(vec.size)

        glpk.glp_set_mat_row(lp, i + 1, vec.size, indices_vec, data_vec)

    return lp

class SwigArray():
    '''
    This is my workaround to fix a memory leak in swig arrays, see: https://github.com/biosustain/swiglpk/issues/31)

    The general idea is to only allocate a single time for each type, and reuse the array
    '''

    dbl_array = []
    dbl_array_size = -1

    int_array = []
    int_array_size = -1

    seq_array = []
    seq_array_size = -1

    @classmethod
    def get_double_array(cls, size):
        'get a double array of the requested size (or greater)'

        if size > cls.dbl_array_size:
            cls.dbl_array_size = 2**math.ceil(math.log(size, 2)) # allocate in multiples of two
            cls.dbl_array = glpk.doubleArray(cls.dbl_array_size)

            #print(f"allocated dbl array of size {cls.dbl_array_size} (requested {size})")

        return cls.dbl_array

    @classmethod
    def get_int_array(cls, size):
        'get a int array of the requested size (or greater)'

        if size > cls.int_array_size:
            cls.int_array_size = 2**math.ceil(math.log(size, 2)) # allocate in multiples of two
            cls.int_array = glpk.intArray(cls.int_array_size)

            #print(f".allocated int array of size {cls.int_array_size} (requested {size})")

        #print(f".returning {cls.int_array} of size {cls.int_array_size} (requested {size})")

        return cls.int_array

    @classmethod
    def as_double_array(cls, list_data, size):
        'wrapper for swig as_doubleArray'

        # about 3x slower than glpk.as_doubleArray, but doesn't leak memory
        arr = cls.get_double_array(size + 1)

        for i, val in enumerate(list_data):
            arr[i+1] = float(val)
            
        return arr

    @classmethod
    def as_int_array(cls, list_data, size):
        'wrapper for swig as_intArray'

        # about 3x slower than glpk.as_intArray, but doesn't leak memory
        arr = cls.get_int_array(size + 1)

        for i, val in enumerate(list_data):
            #print(f"setting {i+1} <- val: {val} ({type(val)}")
            arr[i+1] = val

        return arr

    @classmethod
    def get_sequential_int_array(cls, size):
        'creates or returns a swig int array that counts from 1, 2, 3, 4, .. size'

        if size > (cls.seq_array_size - 1):
            cls.seq_array_size = 1 + 2**math.ceil(math.log(size, 2)) # allocate in multiples of two
            cls.seq_array = glpk.intArray(cls.seq_array_size)

            #print(f"allocated seq array of size {cls.seq_array_size} (requested {size})")

            for i in range(cls.seq_array_size):
                cls.seq_array[i] = i

        return cls.seq_array

def get_num_rows(lp):
    'get the number of rows in the lp'

    return glpk.glp_get_num_rows(lp)

def get_num_cols(lp):
    'get the number of columns in the lp'

    return glpk.glp_get_num_cols(lp)

def get_col_bounds(lp):
    '''get column bounds
    '''

    lp_cols = get_num_cols(lp)

    # column lower and upper bounds
    col_bounds = []

    for col in range(lp_cols):
        col_type = glpk.glp_get_col_type(lp, col + 1)

        ub = np.inf
        lb = -np.inf

        if col_type == glpk.GLP_DB:
            ub = glpk.glp_get_col_ub(lp, col + 1)
            lb = glpk.glp_get_col_lb(lp, col + 1)
        elif col_type == glpk.GLP_LO:
            lb = glpk.glp_get_col_lb(lp, col + 1)
        elif col_type == glpk.GLP_FX:
            lb = ub = glpk.glp_get_col_lb(lp, col + 1)
        else:
            assert col_type == glpk.GLP_FR, "unsupported col type in get_col_bounds()"

        col_bounds.append((lb, ub))

    return col_bounds

def _column_names_str(lp):
    'get the line in __str__ for the column names'

    rv = "    "
    dbl_max = sys.float_info.max

    for col in range(get_num_cols(lp)):
        name = f"x{col}"

        lb = glpk.glp_get_col_lb(lp, col + 1)
        ub = glpk.glp_get_col_ub(lp, col + 1)

        if lb != -dbl_max or ub != dbl_max:
            name = "*" + name

        name = name.rjust(6)[:6] # fix width to exactly 6

        rv += name + " "

    rv += "\n"

    return rv

def _opt_dir_str(lp, zero_print):
    'get the optimization direction line for __str__'

    cols = get_num_cols(lp)
    rv = "min "

    for col in range(1, cols + 1):
        val = glpk.glp_get_obj_coef(lp, col)

        num = f"{val:.6f}"
        num = num.rjust(6)[:6] # fix width to exactly 6

        if val == 0:
            rv += zero_print(num) + " "
        else:
            rv += num + " "

    rv += "\n"

    return rv

def _col_stat_str(lp):
    'get the column statuses line for __str__'

    cols = get_num_cols(lp)

    stat_labels = ["?(0)?", "BS", "NL", "NU", "NF", "NS", "?(6)?"]
    rv = "   "

    for col in range(1, cols + 1):
        rv += "{:>6} ".format(stat_labels[glpk.glp_get_col_stat(lp, col)])

    rv += "\n"

    return rv

def _constraints_str(lp, zero_print):
    'get the constraints matrix lines for __str__'

    rv = ""
    rows = get_num_rows(lp)
    cols = get_num_cols(lp)

    stat_labels = ["?(0)?", "BS", "NL", "NU", "NF", "NS"]
    inds = SwigArray.get_int_array(cols + 1)
    vals = SwigArray.get_double_array(cols + 1)

    for row in range(1, rows + 1):
        stat = glpk.glp_get_row_stat(lp, row)
        assert 0 <= stat <= len(stat_labels)
        rv += "{:2}: {} ".format(row, stat_labels[stat])

        num_inds = glpk.glp_get_mat_row(lp, row, inds, vals)

        for col in range(1, cols + 1):
            val = 0

            for index in range(1, num_inds+1):
                if inds[index] == col:
                    val = vals[index]
                    break

            num = f"{val:.6f}"
            num = num.rjust(6)[:6] # fix width to exactly 6

            rv += (zero_print(num) if val == 0 else num) + " "

        row_type = glpk.glp_get_row_type(lp, row)

        assert row_type == glpk.GLP_UP
        val = glpk.glp_get_row_ub(lp, row)
        rv += " <= "

        num = f"{val:.6f}"
        num = num.rjust(6)[:6] # fix width to exactly 6

        rv += (zero_print(num) if val == 0 else num) + " "

        rv += "\n"

    return rv

def _var_bounds_str(lp):
    'get the variable bounds string used in __str__'

    rv = ""

    dbl_max = sys.float_info.max
    added_label = False

    for index in range(get_num_cols(lp)):
        name = f"x{index}"
        
        lb = glpk.glp_get_col_lb(lp, index + 1)
        ub = glpk.glp_get_col_ub(lp, index + 1)

        if not added_label and (lb != -dbl_max or ub != dbl_max):
            added_label = True
            rv += "(*) Bounded variables:"

        if lb != -dbl_max or ub != dbl_max:
            lb = "-inf" if lb == -dbl_max else lb
            ub = "inf" if ub == dbl_max else ub

            rv += f"\n{name} in [{lb}, {ub}]"

    return rv

def to_str(lp, plain_text=False):
    'get the LP as string (useful for debugging)'

    if plain_text:
        zero_print = lambda x: x
    else:
        def zero_print(s):
            'print function for zeros'

            return colored(s, 'white', attrs=['dark'])

    rows = get_num_rows(lp)
    cols = get_num_cols(lp)
    rv = f"Lp has {cols} columns (variables) and {rows} rows (constraints)\n"

    rv += _column_names_str(lp)

    rv += _opt_dir_str(lp, zero_print)

    rv += "subject to:\n"

    rv += _col_stat_str(lp)

    rv += _constraints_str(lp, zero_print)

    rv += _var_bounds_str(lp)

    return rv

def set_minimize_direction(lp, direction):
    """set the optimization direction"""

    assert len(direction) == get_num_cols(lp), f"expected {get_num_cols(lp)} cols, but optimization " + \
        f"vector had {len(direction)} variables"

    for i, d in enumerate(direction):
        col = int(1 + i)

        glpk.glp_set_obj_coef(lp, col, float(d))
