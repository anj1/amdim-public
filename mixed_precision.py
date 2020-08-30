""" Handles the mixed precision logic """

import torch

MP_CONFIG = {
    'have_apex': False,
    'enabled': False,
    'optimization_level': 'O2'
}

try:
    from apex.fp16_utils import FP16_Optimizer
    MP_CONFIG['have_apex'] = True
except:
    print("nvidia/apex not found; mixed-precision via --apex will be disabled.")
    FP16_Optimizer = None

def enable_mixed_precision():
    if MP_CONFIG['have_apex']:
        MP_CONFIG['enabled'] = True
    else:
        raise RuntimeError()

def is_mixed_precision():
    return MP_CONFIG['have_apex'] and MP_CONFIG['enabled']


def get_optim_level():
    return MP_CONFIG['optimization_level']


def get_optimizer(obj):
    if not MP_CONFIG['have_apex']:
        raise RuntimeError()

    '''
    Apex introduces the FP16_optimizer object.
    However this isn't really an optimizer, but only a wrapper around one.
    This function returns the actual optimizer.
    '''
    if type(obj) == FP16_Optimizer:
        return obj.optimizer
    # If obj is not an FP16_Optimizer then we are not running in mixed precision
    # and the passed object is already an actual optimizer
    return obj


def set_optim_level(opt_level):
    if not MP_CONFIG['have_apex']:
        raise RuntimeError()

    """Defines the optimization level that will be used by AMP
    See: https://nvidia.github.io/apex/amp.html#opt-levels

    Arguments:
        opt_level {string} -- The optimization level to use, should be O1 or O2.
    """
    MP_CONFIG['optimization_level'] = opt_level


def maybe_half(tensor):
    """Convert a tensor to half precision if mixed precision is on.

    Arguments:
        tensor {torch.Tensor} -- The tensor to convert

    Returns:
        torch.Tensor -- returns the converted input tensor
    """
    return tensor.half() if is_mixed_precision() else tensor


def initialize(model, optimizers):
    if not MP_CONFIG['have_apex']:
        raise RuntimeError()
    """Initialize mixed precision

    Arguments:
        model {nn.Module} -- The model to convert
        optimizers -- The model

    Returns:
        [nn.Module, Optimizer] -- Converted model and optimizer
    """
    if is_mixed_precision():
        from apex import amp
        if optimizers is not None:
            model, optimizers = \
                amp.initialize(model, optimizers, opt_level=get_optim_level())
        else:
            model = amp.initialize(model, opt_level=get_optim_level())
    return model, optimizers


def backward(loss, optimizer):
    if not MP_CONFIG['have_apex']:
        raise RuntimeError()
    """Calls backward on the loss. If mixed precision is on, will
    scale the loss.
    """
    if is_mixed_precision():
        from apex import amp
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
