"""
fairness analyzer exploration using nnenum
test the networks

Nov 2021, Stanley Bak
"""

from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import onnx
import onnxruntime as ort

from nnenum.onnx_network import load_onnx_network_optimized, load_onnx_network
from nnenum.network import nn_unflatten

def predict_with_onnxruntime(sess, *inputs):
    'run an onnx model'
    
    names = [i.name for i in sess.get_inputs()]

    inp = dict(zip(names, inputs))
    res = sess.run(None, inp)

    #names = [o.name for o in sess.get_outputs()]

    return res[0]

def init_plot():
    'initialize plotting style'

    #matplotlib.use('TkAgg') # set backend
    plt.style.use(['bmh', 'bak_matplotlib.mlpstyle'])

class RandomSampler:
    """sample randomly from combinations of 1-d distributions"""

    def __init__(self):
        self.ages = [(0, 50), (0.2, 100), (1.0, 0)]
        self.ages_limit = max([a for _, a in self.ages])
        
        self.priors = [(0, 20), (0.2, 5), (1.0, 0)]
        self.priors_limit = max([p for _, p in self.priors])

        self.ages_func = make_linear_interpolation_func(self.ages)
        self.priors_func = make_linear_interpolation_func(self.priors)

    def get_random_pt(self):
        """gets a random point using rejection sampling"""

        age = -1
        prior = -1

        while True:
            age = np.random.rand()
            y = np.random.rand() * self.ages_limit

            if y < self.ages_func(age):
                break

        while True:
            prior = np.random.rand()
            y = np.random.rand() * self.priors_limit

            if y < self.priors_func(prior):
                break

        return np.array([age, prior], dtype=float)

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

    num_samples = int(1e5)
    print(f"sampling {num_samples} times...")

    init_plot()
    onnx_filename = "seed0.onnx"

    # generate random inputs
    sex = "Male"
    network_label = "Seed 0"
    #aam_box = [[0.0, 1.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
    #wm_box = [[0.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 1.0]]

    onnx_model = onnx.load(onnx_filename)
    sess = ort.InferenceSession(onnx_model.SerializeToString())
        
    inp, _out, inp_dtype = get_io_nodes(onnx_model)
    
    inp_shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in inp.type.tensor_type.shape.dim)

    input_34_list = [[1.0, 1.0], [0.0, 1.0]]

    
    lowrisk_lists: List[List[np.ndarray]] = [[], []]
    union_list: List[np.ndarray] = []

    #for color, label, input34 in zip(colors, labels, input_34_list):
    pts_in_union = 0
    sampler = RandomSampler()

    all_pts = []

    for _ in range(num_samples):
        pt = sampler.get_random_pt()
        all_pts.append(pt)

        in_indices = []

        for index, input34 in enumerate(input_34_list):
            sample = np.hstack([pt, input34])

            i = np.array(sample, dtype=inp_dtype)
            i = i.reshape(inp_shape, order='C')

            output = predict_with_onnxruntime(sess, i)[0, 0]

            if output <= 0:
                in_indices.append(index)

        if len(in_indices) == 1:
            lowrisk_lists[in_indices[0]].append(pt)
        elif len(in_indices) == 2:
            union_list.append(pt)
            pts_in_union += 1

    assert all_pts, "all_pts was empty"

    ################## plot 1 - samples #######################
    fig, ax_list = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for ax in ax_list:
        ax.set_xlabel('Age', fontsize=16)
        ax.set_ylabel('Priors', fontsize=16)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    #ax_list[0].set_title('Heatmap', fontsize=16)
    #ax_list[1].set_title('Contour', fontsize=16)

    #ax_list[0].set_title('Rand Input Heatmap')
    #ax_list[1].set_title('Rand Input Contour Plot')
    fig.suptitle("Distribution of Random Inputs", fontsize=24)

    ax_list[0].grid(False)
    counts,ybins,xbins,image = ax_list[0].hist2d(*zip(*all_pts), bins=(30, 30), cmap=plt.cm.rainbow)

    ax_list[1].plot(*zip(*all_pts), '.', color='k', ms=0.03, label="")
    ax_list[1].contour(counts.transpose(), 14,
               extent=[xbins[0],xbins[-1],ybins[0],ybins[-1]], linewidths=3)

    plot_filename = f"rand_{sex}_{network_label.replace(' ', '_')}_inputs.png"
    fig.savefig(plot_filename, bbox_inches='tight')
    print(f"Saved to {plot_filename}")

    plt.clf()

    ############ plot 2 - outputs #############
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f'Random Samples Outputs ({sex}, {network_label})', fontsize=18)

    colors = ['r', 'b', 'lime']
    labels = ['African American', 'White', 'Union']
    
    for pts, color, label in zip(lowrisk_lists + [union_list], colors, labels):
        if pts:
            ax.plot(*zip(*pts), '.', ms=0.03, color=color)
        #ax.plot(*zip(*result_pos), 'x', color=color, label=f"Low Risk {label}")

    for color, label in zip(colors, labels):
        ax.plot([-1], [-1], 'o', color=color, label=f"Classified as Low Risk ({label} Only)")

    ax.set_xlabel('Age')
    ax.set_ylabel('Priors')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    text = ""
    symmetric_difference_pts = 0

    for n in range(2):
        in_pts = len(lowrisk_lists[n]) + len(union_list)
        line = f"Fraction in {labels[n]}: {in_pts/num_samples}"
        text += line + "\n"
        symmetric_difference_pts += in_pts - pts_in_union

    line = f"Fraction in union: {pts_in_union/num_samples}"
    text += line + "\n"
    
    line = f"Fraction in symmetric difference: {symmetric_difference_pts/num_samples}"
    text += line
    
    print(text)
    ax.text(0.3, 0.8, text, va='top')

    ax.legend(loc='upper right')
            
    plot_filename = f"rand_{sex}_{network_label.replace(' ', '_')}_outputs.png"
    fig.savefig(plot_filename, bbox_inches='tight')
    print(f"saved to {plot_filename}")

def get_io_nodes(onnx_model):
    'returns 3 -tuple: input node, output nodes, input dtype'

    sess = ort.InferenceSession(onnx_model.SerializeToString())
    inputs = [i.name for i in sess.get_inputs()]
    assert len(inputs) == 1, f"expected single onnx network input, got: {inputs}"
    input_name = inputs[0]

    outputs = [o.name for o in sess.get_outputs()]
    assert len(outputs) == 1, f"expected single onnx network output, got: {outputs}"
    output_name = outputs[0]

    g = onnx_model.graph
    inp = [n for n in g.input if n.name == input_name][0]
    out = [n for n in g.output if n.name == output_name][0]

    input_type = g.input[0].type.tensor_type.elem_type

    assert input_type in [onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE]

    dtype = np.float32 if input_type == onnx.TensorProto.FLOAT else np.float64

    return inp, out, dtype

if __name__ == "__main__":
    main()
