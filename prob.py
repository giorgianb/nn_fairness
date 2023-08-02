from itertools import product
from scipy.integrate import nquad
import numpy as np
from itertools import chain
from collections import defaultdict
from abc import ABC, abstractmethod
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LinearRegression

class KProbabilityDensity:
    """polynomially computes probability of input at given point"""
    # TODO: add "given", which allows us to predicate on some variable when predicting the 
    # continuous features
    def __init__(self, X, input_class, n_generated_features=10, deg=5, bandwidth=0.1):
        # TODO: check whether we still need this assumption, and if so, ensure it is enforced
        # Assumption: One-Hot indices are contiguous in the array for a one-hot feature
        matches_given = np.apply_along_axis(lambda x: x in input_class, 1, X)
        if np.sum(matches_given) == 0:
            raise ValueError("No instance of class found.")

        X = X[matches_given]

        self.discrete_indices = tuple(sorted(input_class.specification.discrete_indices.values()))
        self.continuous_indices = tuple(sorted(input_class.specification.continuous_indices.values()))
        self.one_hot_indices = tuple(sorted(input_class.specification.one_hot_indices.values()))

        self.discrete = [make_discrete_distribution(X[:, i]) for i in self.discrete_indices]
        self.one_hot = [make_one_hot_distribution(X[:, index_group]) for index_group in self.one_hot_indices]

        self.continuous_bounds = [(np.min(X[:, i]), np.max(X[:, i])) for i in self.continuous_indices]
        self.discrete_bounds = [[(k, k) for k in sorted(d.keys())] for d in self.discrete]
        self.one_hot_bounds = [[(one_hot(k, len(d.keys())), one_hot(k, len(d.keys()))) for k in sorted(d.keys())] for d in self.one_hot]


        self.feature_generator = np.random.normal(size=(deg, n_generated_features, len(self.continuous_indices)))
        self.bandwidth = bandwidth
        self.coeffs, self.bias = self._fit(X)

        
        self.discrete_funcs = [make_discrete_func(d) for d in self.discrete]
        self.one_hot_funcs = [make_one_hot_func(d) for d in self.one_hot]

        # Note we can probably do this analytically
        #self.continuous_volume = nquad(lambda *x: self._continuous_sample(np.array(x)), self.continuous_bounds)[0]
        self.continuous_volume = 1
        self.discrete_volumes = [sum(f(k) for k in d.keys()) for f, d in zip(self.discrete_funcs, self.discrete)]
        self.one_hot_volumes = [sum(f(k) for k in d.keys()) for f, d in zip(self.one_hot_funcs, self.one_hot)]

        regions = sorted(
                chain(
                    zip([[k] for k in self.continuous_indices], self.continuous_bounds),
                    zip([[k] for k in self.discrete_indices], self.discrete_bounds),
                    zip(self.one_hot_indices, self.one_hot_bounds)
                    )

        )
        self._regions = tuple(map(lambda x: x[1], regions))

    def _fit(self, X):
        # Use the kernel to generate the "ground truth" pdf
        X = X[:, self.continuous_indices]
        kern = KernelDensity(kernel='epanechnikov', bandwidth=self.bandwidth).fit(X)
        # TODO: modify X_test so that the bounds given in input_class are respected
        X_test = np.random.uniform(size=(2*X.shape[0], X.shape[1]))
        X_full = np.concatenate((X, X_test), axis=0)
        y = np.exp(kern.score_samples(X_full))

        # Now we generate the features
        X_train = np.einsum('dfi,bi->bdf', self.feature_generator, X_full)
        deg = np.arange(self.feature_generator.shape[0])[None, :, None] + 1
        X_train = X_train**deg
        X_train = X_train.reshape(X_train.shape[0], -1)

        lm = LinearRegression().fit(X_train, y)
        return lm.coef_.reshape(self.feature_generator.shape[:-1]), lm.intercept_

    def _continuous_sample(self, x):
        X = np.einsum('dfi,...i->...df', self.feature_generator, x)
        X = X**(np.arange(self.feature_generator.shape[0])[None, :, None] + 1).reshape(X.shape[0], -1)
        y = np.einsum('...df,df->...', X, self.coeffs) + self.bias
        return y

    def sample(self, *args):
        """get probability density at a point"""

        x = np.array(args)
        p = self._continuous_sample(x[list(self.continuous_indices)])/self.continuous_volume
        for func, volume, index in zip(self.discrete_funcs, self.discrete_volumes, self.discrete_indices):
            p *= func(x[index])/volume

        for func, volume, index_group in zip(self.one_hot_funcs, self.one_hot_volumes, self.one_hot_indices):
            p *= func(x[index_group])/volume

        return p

    @property
    def regions(self):
        return product(*self._regions)

    def integrate(self, lpi, fixed_indices):
        # Fixed indices are those that are fixed in the input set
        # So they are not in the matrix A
        prob = 0

        A_lpi = lpi.get_constraints_csr().toarray()
        b_lpi = lpi.get_rhs()
        lpi_copy = LpInstance(lpi)
       
        for region in self.regions:
            # TODO: move this to a function
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

            if len(to_eliminate_cols) > 0:
                to_eliminate_vals = np.array(to_eliminate_vals)
                b -= A[:, to_eliminate_cols] @ to_eliminate_vals
                A = A[:, to_keep_cols]

            A = np.array(A)
            b = np.array(b)
            poly = reduce(Polytope(A, b))
            poly.minrep = True
            #prob += volume(poly, extreme)*p
            lin_forms = self.feature_generator.reshape(-1, self.feature_generator.shape[-1])
            lin_forms = np.concatenate([lin_forms, np.ones((1, lin_forms.shape[1]))], axis=0)
            P = np.arange(self.feature_generator.shape[0])[None, :].repeat(self.feature_generator.shape[-1], axis=0).flatten()
            P = np.concatenate([P, np.zeros(0)], axis=0)
            v, vertices = lawrence_integrate_polytope(poly, dd_extreme, coeffs=coeffs, P=P)
            v = np.maximum(v, 0)
            coeffs = np.concatenate([self.coeffs, bias*np.ones(1)], dim=-1)
            p += np.sum(v*coeffs)

        return prob



class ProbabilityDensityComputer:
    """computes probability of input at given point"""

    def __init__(self, X, discrete_indices, continuous_indices, one_hot_indices, fixed_indices, class_filter):
        # Assumption: One-Hot indices are contiguous in the array for a one-hot feature
        matches_given = np.apply_along_axis(class_filter, 1, X)
        if np.sum(matches_given) == 0:
            print("warning: no instance of class found.", file=sys.stderr)

        X = X[matches_given]

        self.discrete_indices = tuple(discrete_indices)
        self.continuous_indices = tuple(continuous_indices)
        self.one_hot_indices = tuple(one_hot_indices)
        self.fixed_indices = tuple(fixed_indices)
        self.continuous = [make_continuous_distribution(X[:, i]) for i in continuous_indices]
        self.discrete = [make_discrete_distribution(X[:, i]) for i in discrete_indices]
        self.one_hot = [make_one_hot_distribution(X[:, index_group]) for index_group in one_hot_indices]

        self.continuous_funcs = [make_linear_interpolation_func(d) for d in self.continuous]
        self.discrete_funcs = [make_discrete_func(d) for d in self.discrete]
        self.one_hot_funcs = [make_one_hot_func(d) for d in self.one_hot]

        self.continuous_volumes = [quad(f, d[0][0], d[-1][0], limit=500)[0] for f, d in zip(self.continuous_funcs, self.continuous)]
        self.discrete_volumes = [sum(f(k) for k in d.keys()) for f, d in zip(self.discrete_funcs, self.discrete)]
        self.one_hot_volumes = [sum(f(k) for k in d.keys()) for f, d in zip(self.one_hot_funcs, self.one_hot)]
        self.continuous_bounds = [tuple(zip(d[:, 0], d[1:, 0])) for d in self.continuous]
        self.non_discretized_continuous_bounds = [[(np.min(d[:, 0]), np.max(d[:, 0]))] for d in self.continuous]
        self.discrete_bounds = [[(k, k) for k in sorted(d.keys())] for d in self.discrete]
        self.one_hot_bounds = [[(one_hot(k, len(d.keys())), one_hot(k, len(d.keys()))) for k in sorted(d.keys())] for d in self.one_hot]
        regions = sorted(
                chain(
                    zip([[k] for k in continuous_indices], self.continuous_bounds),
                    zip([[k] for k in discrete_indices], self.discrete_bounds),
                    zip(one_hot_indices, self.one_hot_bounds)
                )
        )

        non_discretized_regions = sorted(
                chain(
                    zip([[k] for k in continuous_indices], self.non_discretized_continuous_bounds),
                    zip([[k] for k in discrete_indices], self.discrete_bounds),
                    zip(one_hot_indices, self.one_hot_bounds)
                    )

        )
        self._regions = tuple(map(lambda x: x[1], regions))
        self._non_discretized_regions = tuple(map(lambda x: x[1], non_discretized_regions))



    def sample(self, *args):
        """get probability density at a point"""

        p = 1
        x = np.array(args)
        for func, volume, index in zip(self.continuous_funcs, self.continuous_volumes, self.continuous_indices):
            p *= func(x[index])/volume

        for func, volume, index in zip(self.discrete_funcs, self.discrete_volumes, self.discrete_indices):
            p *= func(x[index])/volume

        for func, volume, index_group in zip(self.one_hot_funcs, self.one_hot_volumes, self.one_hot_indices):
            p *= func(x[index_group])/volume

        return p

    @property
    def regions(self):
        if hasattr(self, 'discretized'):
            return product(*self._regions)
        return product(*self._non_discretized_regions)

    @property
    def non_discretized_regions(self):
        return product(*self._non_discretized_regions)

def make_poly_interpolation_func(pts, deg):
    """converts a list of 2-d points to an interpolation function
    assumes function is zero outside defined range
    """


    # assuming your points are stored in two lists: x_coords and y_coords
    x_coords = np.array([x for x, y in pts])
    y_coords = np.array([y for x, y in pts])

    # fit the polynomial using the numpy Polynomial class
    p = Polynomial.fit(x_coords, np.sqrt(y_coords), deg)

    def f(x):
        return p(x)**2


    return f



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

def make_continuous_distribution(data):
    counts, boundaries = np.histogram(data, bins=10)
    centers = (boundaries[1:] + boundaries[:-1])/2
    distribution = np.stack((centers, counts), axis=-1)

    return distribution


def make_discrete_distribution(data):
    dist = defaultdict(int)

    for x in data:
        dist[x] += 1

    return dist

def make_one_hot_distribution(data):
    dist = defaultdict(int)

    for x in data:
        hot_index = np.argmax(x == 1)
        dist[hot_index] += 1

    return dist


def make_discrete_func(dist):
    def func(x):
        return dist.get(x, 0)

    return func

def make_one_hot_func(dist):
    def func(x):
        hot_index = np.argmax(x == 1)
        return dist.get(hot_index, 0)

    return func

def one_hot(hot_index, length):
    h = [0]*length
    h[hot_index] = 1
    return tuple(h)



