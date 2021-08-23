from functools import partial

def get_obj(info, module, *args, **kwargs):
    """
    Finds a function handle with the name given as 'type' in info, and returns the
    instance initialized with corresponding arguments given.

    `object = get_obj('name', module, a, b=1)`
    is equivalent to
    `object = module.name(a, b=1)`
    """
    module_name = info['type']
    module_args = dict(info['args'])
    assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    return getattr(module, module_name)(*args, **module_args)

def get_ftn(info, module, *args, **kwargs):
    """
    Finds a function handle with the name given as 'type' in info, and returns the
    function with given arguments fixed with functools.partial.

    `function = get_ftn('name', module, a, b=1)`
    is equivalent to
    `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
    """
    module_name = info['type']
    module_args = dict(info['args'])
    assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
    module_args.update(kwargs)
    return partial(getattr(module, module_name), *args, **module_args)

