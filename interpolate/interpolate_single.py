"""
demonstration code for interpolating 1-d distributations in 2-D

single image version

Stanley Bak
"""

import numpy as np

from scipy.integrate import quad, nquad

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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

def main():
    """main entry point"""

    init_plot()

    ages = [(0, 50), (0.2, 100), (1.0, 0)]
    priors = [(0, 20), (0.2, 5), (1.0, 0)]
    labels = ["Age", "Priors"]

    fig = plt.figure(figsize=(6, 5))

    funcs = [make_linear_interpolation_func(d) for d in [ages, priors]]
    volumes = []

    for i, (label, data, func) in enumerate(zip(labels, [ages, priors], funcs)):
        # 1-d distribution plot
        #ax = plt.subplot2grid((1, 1), (0, i))
        #ax.set_xlabel(label)
        #ax.set_ylabel("Probability")

        #xs = []
        #ys = []

        volume = quad(func, data[0][0], data[-1][0])[0]
        print(f"Volume {label}: {volume}")
        volumes.append(volume)

        #for x, _ in data:
        #    xs.append(x)
        #    ys.append(func(x) / volume)
        
        #ax.plot(xs, ys, 'k-o')

    # volume of 2d plot
    def prod_func(x, y):
        """product of independent distributions"""

        return (funcs[0](x) / volumes[0]) * (funcs[1](y) / volumes[1])

    volume2d = nquad(prod_func, [(0, 1), (0, 1)])[0]

    print(f"volume2d: {volume2d}")

    # contour plot
    ax = plt.subplot2grid((1, 1), (0, 0))

    delta = 0.0025
    X, Y = np.meshgrid(np.arange(0, 1.0 + delta/2, delta), np.arange(0, 1.0 + delta/2, delta))
    z_list = []

    print(X.shape)

    for x_row, y_row in zip(X, Y):
        z_row = []
        z_list.append(z_row)

        for x, y in zip(x_row, y_row):
            z = (funcs[0](x) / volumes[0]) * (funcs[1](y) / volumes[1])
            #z = (funcs[0](x) / volumes[0] + funcs[1](y) / volumes[1]) / 2
            z_row.append(z)

    Z = np.array(z_list, dtype=float)
    cs = ax.contour(X, Y, Z, 14)
    ax.clabel(cs, inline=1, fontsize=10)

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    plt.tight_layout()
    fig.savefig("single.png", bbox_inches='tight')
    #plt.show()
    
if __name__ == "__main__":
    main()
