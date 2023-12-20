import importlib
from easydict import EasyDict


class ParamManager:
    
    def __init__(self, args):
        
        self.args = EasyDict(dict(vars(args)))   
        
def add_config_param(old_args, config_file_name = None):
    
    if config_file_name is None:
        config_file_name = old_args.config_file_name
        
    if config_file_name.endswith('.py'):
        module_name = '.' + config_file_name[:-3]
    else:
        module_name = '.' + config_file_name

    config = importlib.import_module(module_name, 'configs')

    config_param = config.Param
    method_args = config_param(old_args)
    new_args = EasyDict(dict(old_args, **method_args.hyper_param))
    
    return new_args
