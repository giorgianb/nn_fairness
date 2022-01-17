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

def main():
    """main entry point"""

    init_plot()
    onnx_filename = "seed0.onnx"

    # generate random inputs
    sex = "Male"
    network_label = "Seed 0"
    aam_box = [[0.0, 1.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
    wm_box = [[0.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 1.0]]

    onnx_model = onnx.load(onnx_filename)
    sess = ort.InferenceSession(onnx_model.SerializeToString())
        
    inp, out, inp_dtype = get_io_nodes(onnx_model)
    
    inp_shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in inp.type.tensor_type.shape.dim)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f'Random Samples ({network_label})')

    colors = ['b', 'lime']
    labels = [f'African American ({sex})', f'White ({sex})']

    input_34_list = [[1.0, 1.0], [0.0, 1.0]]
    num_samples = 10000

    for color, label, input34 in zip(colors, labels, input_34_list):
        result_neg = []
        result_pos = []

        for _ in range(num_samples):
            pt = np.random.rand(2)
    
    lowrisk_lists: Tuple[List[np.ndarray], List[np.ndarray]] = ([], [])

    #for color, label, input34 in zip(colors, labels, input_34_list):
    pts_in_union = 0

    for _ in range(num_samples):

        while True:
            pt = np.random.rand(2)
            additional_term = np.random.rand()

            if pt[0] + additional_term < 1:
                pt = pt[0:2]
                break

        is_in_union = True

        for index, input34 in enumerate(input_34_list):
            sample = np.hstack([pt, input34])

            i = np.array(sample, dtype=inp_dtype)
            i = i.reshape(inp_shape, order='C')

            output = predict_with_onnxruntime(sess, i)[0, 0]

            if output <= 0:
                result_neg.append(pt)
            else:
                result_pos.append(pt)

        if is_in_union:
            pts_in_union += 1

    for pts, color, label in zip(lowrisk_lists, colors, labels):
        ax.plot(*zip(*pts), '.', color=color, alpha=0.5, label=f"Low Risk {label}")
        #ax.plot(*zip(*result_pos), 'x', color=color, label=f"Low Risk {label}")

    ax.set_xlabel('Age (normalized)')
    ax.set_ylabel('Prior (normalized)')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    ax.legend()
            
    plot_filename = f"{sex}_{network_label.replace(' ', '_')}_samples.png"
    fig.savefig(plot_filename)

    print(f"saved to {plot_filename}")

    print(f"Fraction of points in union: {pts_in_union/num_samples}")
    symmetric_difference_pts = 0

    for n in range(2):
        print(f"Fraction of points in {labels[n]}: {len(lowrisk_lists[n])/num_samples}")
        symmetric_difference_pts += len(lowrisk_lists[n]) - pts_in_union

    print(f"Fraction of points in symmetric difference: {symmetric_difference_pts/num_samples}")

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
