import argparse

import torch
import torch.nn as nn

def count_conv2d(m, x, y):
    x = x[0]

    cin = m.in_channels // m.groups
    cout = m.out_channels // m.groups
    kh, kw = m.kernel_size
    batch_size = x.size()[0]

    # ops per output element
    kernel_mul = kh * kw * cin
    kernel_add = kh * kw * cin - 1
    bias_ops = 1 if m.bias is not None else 0
    ops = kernel_mul + kernel_add + bias_ops

    # total ops
    num_out_elements = y.numel()
    total_ops = num_out_elements * ops

    # incase same conv is used multiple times
    m.total_ops += torch.Tensor([int(total_ops)])

def count_bn2d(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_sub = nelements
    total_div = nelements
    total_ops = total_sub + total_div

    m.total_ops += torch.Tensor([int(total_ops)])

def count_relu(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_ops = nelements

    m.total_ops += torch.Tensor([int(total_ops)])

def count_softmax(m, x, y):
    x = x[0]

    batch_size, nfeatures = x.size()

    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)

    m.total_ops += torch.Tensor([int(total_ops)])

def count_maxpool(m, x, y):
    kernel_ops = torch.prod(torch.Tensor([m.kernel_size])) - 1
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements
    m.total_ops += torch.Tensor([int(total_ops)])

def count_avgpool(m, x, y):
    total_add = torch.prod(torch.Tensor([m.kernel_size])) - 1
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])

def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features
    total_add = m.in_features - 1
    num_elements = y.numel()
    total_ops = (total_mul + total_add) * num_elements
    m.total_ops += torch.Tensor([int(total_ops)])

def count_sigmoid(m, x, y):
    x = x[0]
    nelements = y.numel() 
    total_ops = nelements * 3
    m.total_ops += torch.tensor([int(total_ops)], dtype=torch.float64)

def count_adaptive_avg_pool(m, x, y):
    x = x[0]
    total_ops = x.numel()
    m.total_ops += torch.tensor([int(total_ops)], dtype=torch.float64)

def count_upsample(m, x, y):
    x = x[0]
    nelements = y.numel()
    total_ops = nelements * 7
    m.total_ops += torch.tensor([int(total_ops)], dtype=torch.float64)

def profile(model, input_size, custom_ops={}):
    model.eval()
    handler_collection = [] 

    def add_hooks(m):
        if len(list(m.children())) > 0:
            return 

        # Use float64 for potentially large counts
        m.register_buffer('total_ops', torch.zeros(1, dtype=torch.float64))
        m.register_buffer('total_params', torch.zeros(1, dtype=torch.float64))

        for p in m.parameters():
            m.total_params += torch.tensor([p.numel()], dtype=torch.float64)

        if isinstance(m, nn.Conv2d):
            handler = m.register_forward_hook(count_conv2d)
            handler_collection.append(handler)
        elif isinstance(m, nn.BatchNorm2d):
            handler = m.register_forward_hook(count_bn2d)
            handler_collection.append(handler)
        elif isinstance(m, nn.ReLU):
            handler = m.register_forward_hook(count_relu)
            handler_collection.append(handler)
        elif isinstance(m, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
            handler = m.register_forward_hook(count_maxpool)
            handler_collection.append(handler)
        elif isinstance(m, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
             if hasattr(m, 'kernel_size'):
                  handler = m.register_forward_hook(count_avgpool)
                  handler_collection.append(handler)
             else:
                  print(f"Warning: AvgPool type {type(m)} without fixed kernel_size not implemented.")
        elif isinstance(m, nn.Linear):
            handler = m.register_forward_hook(count_linear)
            handler_collection.append(handler)
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            handler = m.register_forward_hook(count_adaptive_avg_pool)
            handler_collection.append(handler)
        elif isinstance(m, (nn.Upsample, nn.UpsamplingBilinear2d, nn.UpsamplingNearest2d)):
             if m.mode == 'bilinear':
                handler = m.register_forward_hook(count_upsample)
                handler_collection.append(handler)
             else:
                print(f"Warning: OP counting for Upsample mode '{m.mode}' not implemented, skipping.")
                pass
        elif isinstance(m, nn.Sigmoid):
            handler = m.register_forward_hook(count_sigmoid)
            handler_collection.append(handler)
        elif isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            pass 
        else:
            if list(m.parameters()):
                 print(f"Warning: OP counting not implemented for module {type(m)} with parameters.")


    model.apply(add_hooks)

    device = next(model.parameters()).device

    x = torch.zeros(input_size).to(device)

    with torch.no_grad():
        model(x)

    total_ops = torch.zeros(1, dtype=torch.float64)
    total_params = torch.zeros(1, dtype=torch.float64)
    for m in model.modules():
        if hasattr(m, 'total_ops') and len(list(m.children())) == 0:
            total_ops += m.total_ops
        if hasattr(m, 'total_params') and len(list(m.children())) == 0:
             total_params += m.total_params


    for handler in handler_collection:
        handler.remove()
    for m in model.modules():
         if hasattr(m, 'total_ops'):
             del m.total_ops
         if hasattr(m, 'total_params'):
             del m.total_params

    return total_ops.item(), total_params.item()


def main(args):
    model = torch.load(args.model)
    total_ops, total_params = profile(model, args.input_size)
    print("#Ops: %f GOps"%(total_ops/1e9))
    print("#Parameters: %f M"%(total_params/1e6))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pytorch model profiler")
    parser.add_argument("model", help="model to profile")
    parser.add_argument("input_size", nargs='+', type=int,
                        help="input size to the network")
    args = parser.parse_args()
    main(args)