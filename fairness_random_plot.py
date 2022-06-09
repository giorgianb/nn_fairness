

from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import onnx
import onnxruntime as ort

from nnenum.onnx_network import load_onnx_network_optimized, load_onnx_network
from nnenum.network import nn_unflatten
import os
import pickle
import tqdm

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
        cache_path = 'NN-verification/cache'
        random_seed = 0
        cache_file_path = os.path.join(cache_path, f'np-adult-data-rs={random_seed}.pkl')
        with open(cache_file_path, 'rb') as f:
            data_dict = pickle.load(f)

        X_train = data_dict["X_train"]

        self.ages  = make_distribution(X_train[:, 0])
        self.ages_limit = max([a for _, a in self.ages])
        self.edu_number  = make_distribution(X_train[:, 1])
        self.edu_number_limit = max([e for _, e in self.edu_number])
        self.hours_per_week  = make_distribution(X_train[:, 2])
        self.hours_per_week_limit = max([e for _, e in self.hours_per_week])

        self.ages_func = make_linear_interpolation_func(self.ages)
        self.edu_number_func = make_linear_interpolation_func(self.edu_number)
        self.hours_per_week_func = make_linear_interpolation_func(self.hours_per_week)

    def get_random_pt(self):
        """gets a random point using rejection sampling"""

        age = -1
        edu_number = -1
        hours_per_week = -1

        while True:
            age = np.random.rand()
            y = np.random.rand() * self.ages_limit

            if y < self.ages_func(age):
                break

        while True:
            edu_number = np.random.rand()
            y = np.random.rand() * self.edu_number_limit

            if y < self.edu_number_func(edu_number):
                break

        while True:
            hours_per_week = np.random.rand()
            y = np.random.rand() * self.hours_per_week_limit

            if y < self.edu_number_func(hours_per_week):
                break



        return np.array([age, edu_number, hours_per_week], dtype=float)

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

def make_distribution(data):
    counts, boundaries = np.histogram(data)
    centers = (boundaries[1:] + boundaries[:-1])/2
    distribution = np.stack((centers, counts), axis=-1)

    return distribution


def main():
    """main entry point"""

    num_samples = int(1e8)
    print(f"sampling {num_samples} times...")

    onnx_filenames = [
            "NN-verification/results/adult-model_config-small-max_epoch=10-train_bs=32-random_seed=0-is_random_weight-False-race_permute=False-sex_permute=False-both_sex_race_permute=False/model.onnx", 
            "NN-verification/results/adult-model_config-small-max_epoch=10-train_bs=32-random_seed=0-is_random_weight-False-race_permute=True-sex_permute=False-both_sex_race_permute=False/model.onnx", 
            "NN-verification/results/adult-model_config-small-max_epoch=10-train_bs=32-random_seed=0-is_random_weight-False-race_permute=False-sex_permute=True-both_sex_race_permute=False/model.onnx",
            "NN-verification/results/adult-model_config-small-max_epoch=10-train_bs=32-random_seed=0-is_random_weight-False-race_permute=False-sex_permute=False-both_sex_race_permute=True/model.onnx",
            "NN-verification/results/adult-model_config-small-max_epoch=10-train_bs=32-random_seed=0-is_random_weight-True-race_permute=False-sex_permute=False-both_sex_race_permute=False/model.onnx", 
            "NN-verification/results/adult-model_config-medium-max_epoch=10-train_bs=32-random_seed=0-is_random_weight-False-race_permute=False-sex_permute=False-both_sex_race_permute=False/model.onnx", 
            "NN-verification/results/adult-model_config-medium-max_epoch=10-train_bs=32-random_seed=0-is_random_weight-False-race_permute=True-sex_permute=False-both_sex_race_permute=False/model.onnx", 
            "NN-verification/results/adult-model_config-medium-max_epoch=10-train_bs=32-random_seed=0-is_random_weight-False-race_permute=False-sex_permute=True-both_sex_race_permute=False/model.onnx",
            "NN-verification/results/adult-model_config-medium-max_epoch=10-train_bs=32-random_seed=0-is_random_weight-False-race_permute=False-sex_permute=False-both_sex_race_permute=True/model.onnx",
            "NN-verification/results/adult-model_config-medium-max_epoch=10-train_bs=32-random_seed=0-is_random_weight-True-race_permute=False-sex_permute=False-both_sex_race_permute=False/model.onnx", 
    ]


    # generate random inputs
    sex = "Male"
    network_labels = ["(Small) No Permute", "(Small) Race Permute", "(Small) Sex Permute", "(Small) Race & Sex Permute", "(Small) Random Weight", "(Medium) No Permute", "(Medium) Race Permute", "(Medium) Sex Permute", "(Medium) Race & Sex Permute", "(Medium) Random Weight",]
    #aam_box = [[0.0, 1.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
    #wm_box = [[0.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 1.0]]

    for onnx_filename, network_label in zip(onnx_filenames, network_labels):
        print(f"[Testing model '{network_label}']")
        onnx_model = onnx.load(onnx_filename)
        sess = ort.InferenceSession(onnx_model.SerializeToString())
            
        inp, _out, inp_dtype = get_io_nodes(onnx_model)
        
        inp_shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in inp.type.tensor_type.shape.dim)

        # Fills in the rest of the values
        input_rest_list = [[
            1.0,    # is_male
            1.0,    # race_white
            0.0,    # race_black
            0.0,    # race_asian_pac_islander
            0.0,    # race_american_indian_eskimo
            0.0     # race_other
            ], 
            [1.0,   # is_male
            0.0,    # race_white
            1.0,    # race_black
            0.0,    # race_asian_pac_islander
            0.0,    # race_american_indian_eskimo
            0.0     # race_other
            ], 
        ]


        
        lowrisk_lists: List[List[np.ndarray]] = [[], []]
        union_list: List[np.ndarray] = []
        labels = ['White Male', 'Black Male', 'Union']

        #for color, label, rest in zip(colors, labels, input_rest_list):
        pts_in_union = 0
        sampler = RandomSampler()

        all_pts = []

        for _ in tqdm.tqdm(range(num_samples)):
            pt = sampler.get_random_pt()
            all_pts.append(pt)

            in_indices = []

            for index, rest in enumerate(input_rest_list):
                sample = np.hstack([pt, rest])

                i = np.array(sample, dtype=inp_dtype)
                i = i.reshape(inp_shape, order='C')

                output = predict_with_onnxruntime(sess, i)[0, 0]

                if output >= 0:
                    in_indices.append(index)

            if len(in_indices) == 1:
                lowrisk_lists[in_indices[0]].append(pt)
            elif len(in_indices) == 2:
                union_list.append(pt)
                pts_in_union += 1

        assert all_pts, "all_pts was empty"

        symmetric_difference_pts = 0

        print(f"Results for model '{network_label}'")
        for n in range(2):
            in_pts = len(lowrisk_lists[n]) + len(union_list)
            print(f"Fraction in {labels[n]}: {in_pts/num_samples}")
            symmetric_difference_pts += in_pts - pts_in_union

        print(f"Fraction in union: {pts_in_union/num_samples}")
        print(f"Fraction in symmetric difference: {symmetric_difference_pts/num_samples}")

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
