from functools import partial
from omtra.data.graph import get_inv_edge_type
from omtra.data.graph.register import edge_builder_register as register
import functools

@functools.lru_cache()
def get_edge_builders(graph_config):
    edge_builders = {}
    
    edge_configs = graph_config.get('edges', {}).items()
    symmetric_etypes = graph_config.get('symmetric_etypes', [])
    
    for edge_type, config in edge_configs:
        if edge_type == 'npnde_to_npnde':
            continue
        builder_type = config.get('type')
        params = config.get('params', {})
    
        builder_fn = register.get(builder_type)
        if builder_fn is None:
            raise ValueError(f"Edge builder type '{builder_type}' is not registered.")
        
        builder_fn = partial(builder_fn, **params)
        edge_builders[edge_type] = builder_fn
        
        if edge_type in symmetric_etypes:
            inv_edge_type = get_inv_edge_type(edge_type)
            inv_builder_fn = partial(register["symmetric"], builder_fn)
            edge_builders[inv_edge_type] = inv_builder_fn

    return edge_builders
        