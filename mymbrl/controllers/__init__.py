import importlib

def get_item(name):
    module = importlib.import_module("mymbrl.controllers."+name)
    module_class = getattr(module, name)
    return module_class