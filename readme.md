# Neural Decoding Kit (ndkit)

Decoding invasive neural signals **reliably and efficiently** into behavioral variables is a fundamental challenge in brain–computer interfaces (BCIs) and computational neuroscience.

`ndkit` is a **PyTorch-based neural decoding toolkit** that integrates a wide range of state-of-the-art deep learning models. It provides a **reproducible, extensible, and unified** framework for training and evaluating neural decoders.

With `ndkit`, you can:

- **Quickly build** neural decoders  
- **Systematically compare** model performance across datasets and tasks
- **Easily extend** the system with custom models or datasets



# 1. Supported Models

ndkit implements a broad collection of neural decoding models, this section lists all currently supported models.

* [x] **WF** — wiener filter (linear regression)
* [x] **KF** — kalman filter regression
* [x] **FFN** — feedforward neural network
* [x] **RNN** — vanilla recurrent neural network
* [x] **GRU** — gated recurrent unit
* [x] **LSTM** — long short-term memory
* [x] **Transformer** — standard transformer
* [x] **iTransformer** — inverted transformer for time series
* [x] **RWKV** — time-mix + channel-mix hybrid RNN
* [x] **Mamba** — selective state space model
* [x] **DyEnsemble** — dynamically assembled state-dependent decoder
* [x] **StateMoE** — nonlinear state-dependent decoder



# 2. Setup

## 2.1 Environment Setup

`ndkit` requires **Python 3.10**.

### Create a Conda environment

```bash
conda create -n ndkit python=3.10 -y
conda activate ndkit
````

### Install dependencies

```bash
pip install -r requirements.txt
```

## 2.2 Data Preparation

This toolkit uses datasets from the **Neural Latents Benchmark**:
🔗 [https://neurallatents.github.io/](https://neurallatents.github.io/)

We follow the preprocessing format from:
🔗 [https://github.com/seanmperkins/bci-decoders](https://github.com/seanmperkins/bci-decoders)

Place the processed datasets under:

```
data/nlb_data/
```

or update paths in your configuration file.



# 3. Usage

## 3.1 Quick Start

### 3.1.1 Train a model

Run training using a configuration file:

```bash
python main.py -m train -c configs/NLB-GRU.yaml
```

Or override YAML parameters from the command line:

```bash
python main.py -m train -c configs/NLB-GRU.yaml train.n_epochs=2 model.
```

### 3.1.2 Evaluate a model

Evaluate a trained model on the testset by loading a checkpoint:

```bash
python main.py -m eval -c configs/NLB-GRU.yaml -k path/to/best_model.pt
```

### 3.1.3 Train and then evaluate automatically

```bash
python main.py -m train_eval -c configs/NLB-GRU.yaml
```

This runs:

1. Training
2. Saving the best checkpoint
3. Evaluating on the test set

## 3.2 Extensions

### 3.2.1 Custom Models

To add a model, create a file ending with `Model.py` under:

```
ndkit/models/
```

Your model must:

* Be registered with `@register_model("YourModelName")`
* Accept a single argument `cfg` in its constructor

**Example:**

```python
@register_model("YourModelName")
class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_size = getattr(cfg, "input_size") # required field
        self.hidden_size = getattr(cfg, "hidden_size", 128)  # optional
```

### 3.2.2 Custom Datasets

To add a dataset, create a file ending with `Dataset.py` under:

```
ndkit/datasets/
```

Your dataset must:

* Be registered with `@register_dataset("YourDatasetName")`
* Accept `(cfg, split)` in its constructor

**Example:**

```python
@register_dataset("YourDatasetName")
class MyDataset(Dataset):
    def __init__(self, cfg, split):
        ...
```

### 3.2.3 Create Configs

To use a custom model or dataset, create a corresponding configuration file under:

```
configs/
```

The names in the config file must match the registered names:

```yaml
data:
  name: YourDatasetName
  ...

model:
  name: YourModelName
  ...
```

> **About `model.runner_type`:**  
> Use `runner_type: trainloop` for PyTorch models trained with epochs and backpropagation (this applies to most deep learning models).  
> Use `runner_type: fit` for models that provide their own `fit(X, Y)` method (e.g., WF, KF, DyEnsemble).


# Acknowledgments

This project is developed by the **Brain and Brain-Inspired Computing (BBIC) Lab**.

Special thanks to the following open-source projects:

* [Neural Latents Benchmark](https://neurallatents.github.io/)
* [KordingLab/Neural_Decoding](https://github.com/KordingLab/Neural_Decoding)
* [seanmperkins/bci-decoders](https://github.com/seanmperkins/bci-decoders)
* [SynergyNet](https://github.com/lizlive/SynergyNet)
* [Time-Series-Library](https://github.com/thuml/Time-Series-Library)