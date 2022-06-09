

from itertools import chain

import numpy as np
import onnx
import onnxruntime as ort

from nnenum.onnx_network import load_onnx_network_optimized, load_onnx_network
from nnenum.network import nn_unflatten

def predict_with_onnxruntime(model_def, *inputs):
    'run an onnx model'
    
    sess = ort.InferenceSession(model_def.SerializeToString())
    names = [i.name for i in sess.get_inputs()]

    inp = dict(zip(names, inputs))
    res = sess.run(None, inp)

    #names = [o.name for o in sess.get_outputs()]

    return res[0]

def main():
    """main entry point"""

    onnx_filename = "seed0.onnx"

    in_1 = [0.2580645, 0., 0., 1.]
    out_1 = [-2.2498105]

    in_2 = [0.12903225, 0.8888889, 1., 1.]
    out_2 = [4.02982]

    ins = [in_1, in_2]
    outs = [out_1, out_2]

    test_ort(onnx_filename, ins, outs)
    print()
    test_nnenum_execute(onnx_filename, ins, outs)

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

def test_ort(onnx_filename, ins, outs):
    """test with onnx runtime"""

    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model, full_check=True)

    inp, out, inp_dtype = get_io_nodes(onnx_model)
    
    inp_shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in inp.type.tensor_type.shape.dim)
    out_shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in out.type.tensor_type.shape.dim)

    print(f"inp_shape: {inp_shape}")
    print(f"out_shape: {out_shape}")
    print(f"inp_dtype: {inp_dtype}")

    for i, o in zip(ins, outs):
        input_tensor = np.array(i, dtype=inp_dtype)
        input_tensor = input_tensor.reshape(inp_shape, order='C')
        print(f"input_tensor: {input_tensor}")

        output = predict_with_onnxruntime(onnx_model, input_tensor)

        print(output)
        assert np.allclose(output, o)

    print("ort tests passed.")

def test_nnenum_execute(onnx_filename, ins, outs):
    """test with nnenum data structures"""

    network = load_onnx_network_optimized(onnx_filename)

    for i, o in zip(ins, outs):

        input_tensor = np.array(i)
        input_tensor = nn_unflatten(input_tensor, network.get_input_shape())
        out = network.execute(input_tensor)

        print(out)
        assert np.allclose(out, o)

    print("nnenum execute() tests passed.")

if __name__ == "__main__":
    main()
