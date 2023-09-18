import importlib

def get_item(name):
    dict = {
        "free_trend": "FreeTrend"
    }
    module = importlib.import_module("mymbrl.dataloaders."+name)
    module_class = getattr(module, dict[name])
    return module_class