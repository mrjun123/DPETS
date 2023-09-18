import importlib

def get_item(name):
    dict = {
        "MBRL": "mbrl"
    }
    module = importlib.import_module("mymbrl.experiments."+dict[name])
    module_class = getattr(module, name)
    return module_class
    