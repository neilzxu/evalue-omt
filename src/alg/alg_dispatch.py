_ALG_DISPATCH = {}


def add_alg(name):
    def add_to_dispatch(fn):
        _ALG_DISPATCH[name] = fn
        return fn

    return add_to_dispatch


def get_alg(name):
    return _ALG_DISPATCH[name]


def list_algs():
    return _ALG_DISPATCH.keys()
