from mithril.cores.python.numpy.ops import *
from mithril.cores.python.numpy.ops_grad import *
make_array = partial(make_array, default_dtype='float32')

def evaluate(params, data, cache):
    _dtype = cache['dtype']
    freqs_cis = data['freqs_cis']
    input = data['input']
    weight_0 = params['weight_0']
    weight_1 = params['weight_1']
    weight_2 = params['weight_2']
    weight_3 = params['weight_3']
    output_0 = transpose(weight_0, None)
    queries = make_array(matrix_multiplication(input, output_0))
    del output_0
    output_1 = transpose(weight_1, None)
    keys = make_array(matrix_multiplication(input, output_1))
    del output_1
    output_2 = transpose(weight_2, None)
    values = make_array(matrix_multiplication(input, output_2))
    del output_2
    output_8 = reshape(keys, (2, 16, 4, -1))
    del keys
    output_9 = transpose(output_8, (0, 2, 1, 3))
    del output_8
    output_11 = reshape(output_9, (2, 4, 1, 16, -1))
    del output_9
    keys_repeated = make_array(concat(output_11, output_11, axis=2))
    del output_11
    output_13 = reshape(values, (2, 16, 4, -1))
    del values
    output_14 = transpose(output_13, (0, 2, 1, 3))
    del output_13
    output_16 = reshape(output_14, (2, 4, 1, 16, -1))
    del output_14
    values_repeated = make_array(concat(output_16, output_16, axis=2))
    del output_16
    output_18 = reshape(queries, (2, 16, 8, -1))
    del queries
    output_19 = transpose(output_18, (0, 2, 1, 3))
    del output_18
    output_21 = reshape(keys_repeated, (2, 8, 16, -1))
    xq_ = reshape(output_19, (2, 8, 16, 32, 2))
    del output_19
    xk_ = reshape(output_21, (2, 8, 16, 32, 2))
    del output_21
    freqs_split = split(freqs_cis, 2, -1)
    output_33 = make_array(indexer(freqs_split, 0))
    freqs_cos = reshape(output_33, (1, 1, 16, 32, 1))
    del output_33
    output_35 = make_array(indexer(freqs_split, 1))
    del freqs_split
    freqs_sin = reshape(output_35, (1, 1, 16, 32, 1))
    del output_35
    xq_split = split(xq_, 2, -1)
    del xq_
    output_37 = indexer(xq_split, 0)
    cos_xq_real = make_array(multiplication(freqs_cos, output_37))
    output_38 = indexer(xq_split, 1)
    del xq_split
    sin_xq_imag = make_array(multiplication(freqs_sin, output_38))
    xq_out_real = make_array(subtract(cos_xq_real, sin_xq_imag))
    del cos_xq_real
    del sin_xq_imag
    sin_xq_real = make_array(multiplication(freqs_sin, output_37))
    del output_37
    cos_xq_imag = make_array(multiplication(freqs_cos, output_38))
    del output_38
    xq_out_imag = make_array(add(sin_xq_real, cos_xq_imag))
    del sin_xq_real
    del cos_xq_imag
    xq_out_combined = make_array(concat(xq_out_real, xq_out_imag, axis=-1))
    del xq_out_real
    del xq_out_imag
    xq_out = reshape(xq_out_combined, (2, 8, 16, 64))
    del xq_out_combined
    xk_split = split(xk_, 2, -1)
    del xk_
    output_40 = indexer(xk_split, 0)
    cos_xk_real = make_array(multiplication(freqs_cos, output_40))
    output_41 = indexer(xk_split, 1)
    del xk_split
    sin_xk_imag = make_array(multiplication(freqs_sin, output_41))
    xk_out_real = make_array(subtract(cos_xk_real, sin_xk_imag))
    del cos_xk_real
    del sin_xk_imag
    sin_xk_real = make_array(multiplication(freqs_sin, output_40))
    del freqs_sin
    del output_40
    cos_xk_imag = make_array(multiplication(freqs_cos, output_41))
    del freqs_cos
    del output_41
    xk_out_imag = make_array(add(sin_xk_real, cos_xk_imag))
    del sin_xk_real
    del cos_xk_imag
    xk_out_combined = make_array(concat(xk_out_real, xk_out_imag, axis=-1))
    del xk_out_real
    del xk_out_imag
    xk_out = reshape(xk_out_combined, (2, 8, 16, 64))
    del xk_out_combined
    output_43 = multiplication(xq_out, 0.125)
    del xq_out
    output_44 = transpose(xk_out, (0, 1, 3, 2))
    output_45 = make_array(matrix_multiplication(output_43, output_44))
    del output_43
    del output_44
    output_46 = cast(output_45, _dtype)
    attention_weights = softmax(output_46)
    del output_46
    output_47 = dtype(output_45)
    del output_45
    output_48 = cast(attention_weights, output_47)
    del attention_weights
    del output_47
    output_50 = reshape(values_repeated, (2, 8, 16, -1))
    output_51 = make_array(matrix_multiplication(output_48, output_50))
    del output_48
    output_52 = transpose(output_51, (0, 2, 1, 3))
    del output_51
    output_54 = reshape(output_52, (2, 16, -1))
    del output_52
    output_55 = transpose(weight_3, None)
    output = make_array(matrix_multiplication(output_54, output_55))
    del output_54
    del output_55
    return {'keys_out': xk_out, 'keys_repeated': keys_repeated, 'output': output, 'values_out': output_50, 'values_repeated': values_repeated}