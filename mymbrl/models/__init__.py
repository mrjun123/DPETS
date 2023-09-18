import importlib

def get_item(name):
    dict = {
        "cartpole_model": "CartpoleModel"
    }
    module = importlib.import_module("mymbrl.models."+name)
    module_class = getattr(module, dict[name])
    return module_class
    