from torch.utils.tensorboard import SummaryWriter
import faulthandler
faulthandler.enable()
from dotmap import DotMap
import mymbrl.experiments as experiments

from mymbrl.utils import new_run_directory, merge_dict
import os
import pprint
import numpy as np
import torch
import random
import argparse
import json
import optuna
# import paddle

def main(config):

    run_dir_path = 'runs/mymbrl'
    if config_name != 'custom':
            run_dir_path = 'runs/'+config_name+'/mymbrl'
    
    run_dir = new_run_directory(run_dir_path)
    writer = SummaryWriter(run_dir)

    writer.add_text("config", json.dumps(config))
    # writer.add_hparams(config, config)
    
    # 写入配置
    with open(os.path.join(run_dir, "config.txt"), "w") as f:
        f.write(pprint.pformat(config))
    config = DotMap(config)
    config.run_dir = run_dir

    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    # paddle.seed(config.random_seed)
    random.seed(config.random_seed)
    print(config)

    experiment_name = config.experiment.name
    experiment = experiments.get_item(experiment_name)
    run_experiment = experiment(config, writer)
    exp_reward = run_experiment.run()
    return exp_reward

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='This is mymbrl.')
    parser.add_argument('--config', type=str, default='custom')
    parser.add_argument('--noise', type=float, default=0)
    parser.add_argument('--seed', type=int, default=-1)

    args = parser.parse_args()
    config_name = args.config
    noise = args.noise
    seed = args.seed
    
    with open("configs/"+config_name+".json", 'r', encoding='UTF-8') as f:
        config = json.loads(f.read())
    with open("default_config.json", 'r', encoding='UTF-8') as f:
        default_config = json.loads(f.read())

    config = merge_dict(default_config, config)

    if noise > 0:
        config['experiment']['noise'] = noise
    if seed != -1:
        config['random_seed'] = seed

    main(config)