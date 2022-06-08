

# pip3 install polytope

import time

import numpy as np
from scipy.spatial import ConvexHull

from polytope import Polytope, extreme

from compute_volume import quad_integrate_polytope, rand_integrate_polytope

def test_2d_volume():
    """test 2d volume"""

    # x >= -1
    # y >= 0
    # x + y <= 1
    # corners should be (-1,0), (1, 0), (-1, 2)
    # volume should be 2.0

    A = np.array([[0, -1], [-1, 0], [1, 1]], dtype=float)
    b = np.array([1, 0, 1], dtype=float)

    start = time.perf_counter()
    poly = Polytope(A, b)
    vertices = extreme(poly)
    hull = ConvexHull(vertices)
    qhull_time = time.perf_counter() - start

    print(f"qhull volume: {hull.volume}; time: {round(qhull_time * 1000, 3)}ms")

    start = time.perf_counter()
    v = quad_integrate_polytope(A, b)
    quad_time = time.perf_counter() - start
    
    print(f"quad volume: {v}; time: {round(quad_time * 1000, 3)}ms")
    
    # test random
    for samples in [1000, 10000, 100000]:
        start = time.perf_counter()
        rand_vol = rand_integrate_polytope(A, b, samples=samples)
        rand_time = time.perf_counter() - start
        
        print(f"with {samples} rand samples, volume: {rand_vol}; time: {round(rand_time * 1000, 3)}ms")

def test_4d_volume():
    """test 4d volume"""

    # x >= -1
    # y >= 0
    # z >= 0
    # w >= 0
    # x + y <= 1
    # x + 2y + z <= 1
    # 2*x + 2*y + 2*z + w <= 1

    A = np.array([[0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1],
                  [1, 1, 0, 0], [1, 2, 1, 0], [2, 2, 2, 1]], dtype=float)
    b = np.array([1, 0, 0, 0, 1, 1, 1], dtype=float)

    start = time.perf_counter()
    poly = Polytope(A, b)
    vertices = extreme(poly)
    hull = ConvexHull(vertices)
    qhull_time = time.perf_counter() - start

    print(f"qhull volume: {hull.volume}; time: {round(qhull_time * 1000, 3)}ms")

    start = time.perf_counter()
    quad_vol = quad_integrate_polytope(A, b)
    quad_time = time.perf_counter() - start
    
    print(f"quad volume: {quad_vol}; time: {round(quad_time * 1000, 3)}ms")
    
    # test random
    for samples in [1000, 10000, 100000]:
        start = time.perf_counter()
        rand_vol = rand_integrate_polytope(A, b, samples=samples)
        rand_time = time.perf_counter() - start
        
        print(f"with {samples} rand samples, volume: {rand_vol}; time: {round(rand_time * 1000, 3)}ms")

def test_4d_integrate():
    """test 4d integrate. same as volume but doing f(x,y,z,w) = 1 + x*2y*2z*w"""

    # x >= -1
    # y >= 0
    # z >= 0
    # w >= 0
    # x + y <= 1
    # x + 2y + z <= 1
    # x + y + z + w <= 1

    A = np.array([[0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1],
                  [1, 1, 0, 0], [1, 2, 1, 0], [1, 1, 1, 1]], dtype=float)
    b = np.array([1, 0, 0, 0, 1, 1, 1], dtype=float)

    def integ_func(x, y, z, w):
        """function to be integrated over polytope domain"""

        #print(x)
        return 1+ x * 2*y + 2*z * w

    start = time.perf_counter()
    v = quad_integrate_polytope(A, b, integ_func)
    quad_time = time.perf_counter() - start
    
    print(f"quad volume: {v}; time: {round(quad_time * 1000, 3)}ms")
    
    # test random
    for samples in [1e3, 1e4, 1e5]:
        start = time.perf_counter()
        rand_vol = rand_integrate_polytope(A, b, integ_func, int(samples))
        rand_time = time.perf_counter() - start
        
        print(f"with {samples} rand samples, volume: {rand_vol}; time: {round(rand_time * 1000, 3)}ms")

def main():
    """main entry point"""

    print("2d test")
    test_2d_volume()

    print("\n4d test")
    test_4d_volume()

    print("\n4d integrate test")
    test_4d_integrate()

if __name__ == "__main__":
    main()
