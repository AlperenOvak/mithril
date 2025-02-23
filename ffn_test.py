from mithril.cores.python.numpy.ops import *
from mithril.cores.python.numpy.ops_grad import *
make_array = partial(make_array, default_dtype='float32')

def evaluate(params, data, cache):
    input = data['input']
    left = cache['left']
    weight_0 = params['weight_0']
    weight_1 = params['weight_1']
    weight_2 = params['weight_2']
    output_0 = transpose(weight_0, None)
    w1_out = make_array(matrix_multiplication(input, output_0))
    del output_0
    output_1 = transpose(weight_1, None)
    w3_out = make_array(matrix_multiplication(input, output_1))
    del output_1
    _minus = minus(w1_out)
    _exp = exp(_minus)
    del _minus
    _add = make_array(add(left, _exp))
    del _exp
    silu_out = divide(w1_out, _add)
    del w1_out
    del _add
    multiplied = multiplication(silu_out, w3_out)
    del silu_out
    del w3_out
    output_2 = transpose(weight_2, None)
    output = matrix_multiplication(multiplied, output_2)
    del multiplied
    del output_2
    return {'output': output}