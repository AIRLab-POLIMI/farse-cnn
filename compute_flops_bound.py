import math
import yaml
import argparse

def active_rf_bound(n, k, s):
    n = math.sqrt(n)  # 1-d n, square kernels are assumed
    if loose_bound:
        return active_rf_bound_loose(n, k)**2
    else:
        return active_rf_bound_tight(n, k, s)**2

def active_rf_bound_tight(n, k, s):
    return math.floor((k+n-2)/s) + 1

def active_rf_bound_loose(n, k):
    return n + (k-1)

def farseconv_flops(n, i, h, k, s):
    N_l = active_rf_bound(n, k, s)
    return N_l * 4*h*(2*(h+i) + min(n,k*k)), N_l

def sparsepool_flops(n, alpha, k, s):
    N_l = active_rf_bound(n, k, s)
    if k==s:
        f = n * alpha
    else:
        f = N_l * min(n,k*k) * alpha
    return f, N_l

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/model/farsecnn_NCars.yaml",
                        help="path to the config of the model to be evaluated")
    parser.add_argument("--loose", action='store_true',
                        help="use the looser bound for the number of active receptive fields")
    args = parser.parse_args()
    loose_bound = args.loose
    config_path = args.cfg
    print("Computing the FLOPS/ev for model defined at ",config_path)
    if loose_bound:
        print("--loose flag was provided, the loose bound is considered for the number of active receptive fields.")
    print('\n')

    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


    i = len(config['feature_mode'])
    n = 1
    amortized_factor = 1
    flops = 0

    asynsparse_stack = config['farsecnn_layers']
    for l in asynsparse_stack:
        module_name = l['name']
        if module_name == 'FARSEConv':
            k = l['kernel_size']
            s = l['stride']
            h = l['hidden_size']
            f, active_rf = farseconv_flops(n, i, h, k, s)
            i = h
            n = active_rf
        elif module_name == 'SubmanifoldFARSEConv':
            k = l['kernel_size']
            h = l['hidden_size']
            f, active_rf = farseconv_flops(n, i, h, k, 1)
            i = h
        elif module_name == 'SparseMaxPool' or module_name == 'SparseAvgPool':
            k = l['kernel_size']
            s = l.get('stride', k)
            if module_name == 'SparseMaxPool':
                alpha = i
            if module_name == 'SparseAvgPool':
                alpha = 2*i
            f, active_rf = sparsepool_flops(n, alpha, k, s)
            n = active_rf
        elif module_name == 'TemporalDropout':
            window_size = l['window_size']
            amortized_factor = amortized_factor*(1/window_size)
            f = 0
        elif module_name == 'SparseAdaptiveAvgPool' or module_name == 'SparseAdaptiveMaxPool':
            # cost computation of adaptive pooling layers not supported
            f = 0
            n = l['output_size']

        flops = flops + f * amortized_factor
        print(module_name)
        print('\tActive RF bound:',active_rf)
        print('\tNon-amortized module FLOPs:',f/1e6,'M')
        print('\tAmortization factor',amortized_factor)
        print('\tTotal FLOPs:',flops/1e6,'M')
    print("Total model FLOPs", flops/1e6,'M')