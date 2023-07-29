from itertools import product
from scipy.integrate import quad
import numpy as np
from itertools import chain
from collections import defaultdict
from abc import ABC, abstractmethod
from sklearn.neighbors import KernelDensity

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

        self.discrete_indices = tuple(sorted(input_class.discrete_indices.values()))
        self.continuous_indices = tuple(sorted(input_class.continuous_indices.values()))
        self.one_hot_indices = tuple(sorted(input_class.one_hot_indices.values()))

        self.discrete = [make_discrete_distribution(X[:, i]) for i in discrete_indices]
        self.one_hot = [make_one_hot_distribution(X[:, index_group]) for index_group in one_hot_indices]

        self.continuous_bounds = [(np.min(X[:, i]), np.max(X[:, i])) for i in continuous_indices]
        self.discrete_bounds = [[(k, k) for k in sorted(d.keys())] for d in self.discrete]
        self.one_hot_bounds = [[(one_hot(k, len(d.keys())), one_hot(k, len(d.keys()))) for k in sorted(d.keys())] for d in self.one_hot]


        self.feature_generator = np.random.normal(size=(deg, n_generated_features, len(self.continuous_indices)))
        self.bandwidth = bandwidth
        self.coeffs, self.bias = self._fit(X)

        
        self.discrete_funcs = [make_discrete_func(d) for d in self.discrete]
        self.one_hot_funcs = [make_one_hot_func(d) for d in self.one_hot]

        self.continuous_volume = nquad(self._continuous_sample, self.continuous_bounds)
        self.discrete_volumes = [sum(f(k) for k in d.keys()) for f, d in zip(self.discrete_funcs, self.discrete)]
        self.one_hot_volumes = [sum(f(k) for k in d.keys()) for f, d in zip(self.one_hot_funcs, self.one_hot)]

        regions = sorted(
                chain(
                    zip([[k] for k in continuous_indices], self.continuous_bounds),
                    zip([[k] for k in discrete_indices], self.discrete_bounds),
                    zip(one_hot_indices, self.one_hot_bounds)
                    )

        )
        self._regions = tuple(map(lambda x: x[1], regions))

    def _fit(X):
        X_train = np.einsum('dfi,...i->...df', self.feature_generator, X[:, self.continuous_indices])
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = np.random.uniform(size=(2*len(X), len(self.continuous_indices)))
        kern = KernelDensity(kernel='epanechnikov', bandwidth=self.bandwidth).fit(X_train)
        X_full = np.concatenate((X_train, X_test), axis=0)
        y = np.exp(kern.score_samples(X_full))
        lm = LinearRegression(positive=True).fit(X_full, y)
        return lm.coef_.reshape(self.feature_generator.shape[:-1]), lm.intercept_

    def _continuous_sample(self, x):
        X = np.einsum('dfi,..i->...df', self.feature_generator, x)
        y = np.einsum('...df,df->...', X, self.coeffs) + self.bias
        return y

    def sample(self, *args):
        """get probability density at a point"""

        x = np.array(args)
        p = self._continuous_sample(x)/self.continuous_volume
        for func, volume, index in zip(self.discrete_funcs, self.discrete_volumes, self.discrete_indices):
            p *= func(x[index])/volume

        for func, volume, index_group in zip(self.one_hot_funcs, self.one_hot_volumes, self.one_hot_indices):
            p *= func(x[index_group])/volume

        return p

    @property
    def regions(self):
        return product(*self._regions)


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



