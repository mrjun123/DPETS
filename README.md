# Dropout-based Probabilistic Ensembles with Trajectory Sampling

Code to reproduce the experiments in [Practical Probabilistic Model-based Deep Reinforcement Learning by Integrating Dropout Uncertainty and Trajectory Sampling](). This paper is currently submitted to IEEE Transactions on Neural Networks and Learning Systems (TNNLS) for peer review.

![Snipaste_2023-09-20_13-56-43](https://markdown-1259473343.cos.ap-beijing.myqcloud.com/uPic/Snipaste_2023-09-20_13-56-43.png)

## Requirements

1. Install [MuJoCo 1.31](https://www.roboti.us/download.html) and copy your [license](https://www.roboti.us/file/mjkey.txt) key to `~/.mujoco/mjkey.txt`. 
2. Install [PyTorch](https://pytorch.org/get-started/previous-versions/), we recommend CUDA 11.6 and Pytorch 1.13.1
3. Other dependencies can be installed with `pip install -r requirements.txt`.

## Running Experiments

Experiment for a specific configuration can be run using:

```python
python main.py --config cartpole
```

The specific configuration file is located in the `configs` directory and the default configuration file can be located in the root directory `default_ config.json` was found, which allows you to modify the experimental parameters.

## Logging

We use Tensorboard to record experimental data, you can view runs with:

```python
tensorboard --logdir ./ --port=6006 --host=0.0.0.0
```

## Reference

Loading..

