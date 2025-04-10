edge_builder_register = {}

def register_edge_builder(name: str):
    def decorator(fn):
        fn.name = name
        edge_builder_register[name] = fn
        return fn
    return decorator

def get_edge_builder(name: str):
    return edge_builder_register[name]