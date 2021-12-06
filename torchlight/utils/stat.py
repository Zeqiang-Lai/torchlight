def print_model_size_caffe(prototxt_path):
    import caffe
    caffe.set_mode_cpu()
    from numpy import prod, sum
    from pprint import pprint

    print("Net: " + prototxt_path)
    net = caffe.Net(prototxt_path, caffe.TEST)
    print("Layer-wise parameters: ")
    pprint([(k, v[0].data.shape) for k, v in net.params.items()])
    size = sum([prod(v[0].data.shape) for k, v in net.params.items()]) / 1e6
    print(f"Total number of parameters: {size:.2f}M")
    return size

def print_model_size_torch(model):
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))
