import importlib

def get_item(name):
    dict = {
        "dpets": "DPETS"
    }
    module = importlib.import_module("mymbrl.agents."+name)
    module_class = getattr(module, dict[name])
    return module_class