# EEG Classification Baselines

The goal of this project is to establish a set of reproducible baseline results for EEG signal classification.

## Dataset

This project is designed to work with *HMC* and *TUSZ*. 

In this project, we process the dataset into the following structure.

```bash
# ./data
├── HMC
│    ├── train
│    │    ├── sample.npy
│    │    └── ...
│    ├── dev # Same as train
│    ├── eval # Same as train
│    ├── train.json
│    ├── dev.json
│    └── eval.json
└── TUSZ # Same as HMC
```

And the JSON file is in the following format, this is part from `/data/HMC/eval.json`.

```json
[
    {
        "id": "HMC_SN131_seg606_ma",
        "label": 2
    },
    {
        "id": "HMC_SN131_seg607_ma",
        "label": 2
    },
]
```

<!-- how to construct the dataset -->

### *HMC*

<!-- Dataset information -->

In the JSON file, label 0, 1, 2, 3 stands for *Wake*, *NERM1*, *NERM2* and *NERM3* respectively.

We use the EEGs of the first 100 subjects as train set, middle 25 subjects as dev set and last 26 subjects as eval set.

| Split | Wake | NERM1 | NERM2 | NERM3 |
|-|-|-|-|-|
| Train | 47085 | 31632 | 100404 | 51000 |
| Dev | 12558 | 6408 | 23316 | 14883 |
| Eval | 11415 | 8604 | 26529 | 14130 |

### *TUSZ*

<!-- Dataset information -->

In the JSON file, label 0 stands for no seizure shown, 1 for seizure.

| Split | Not seizure | Seizure |
|-|-|-|
| Train | 295320 | 19301 |
| Dev | 82441 | 5707 |
| Eval | 42931 | 3160 |

<!-- Dataset link -->

## Models

<a id = 'EEGNet'></a>
### EEGNet

A compact convolutional neural network for EEG-based brain-computer interfaces.

Set the parameters as in the [Paper](https://arxiv.org/pdf/1611.08024).

![EEGNet Structure](./fig/eegnet_structure.png)

### CNN+Transformers

A CNN+Transformer architecture is a common baseline that appears in many papers on EEG models, see [Paper](https://arxiv.org/pdf/2208.02405) for detailed structure.

Since the *Segment-level* model in the paper is too complex. We make a change to *Channel-level* model so that it can handle multi-channel data: Apply channel-level convolution for each channel, then concantate them together as the input of the transfomer encoder. 

Below is the structure of the channel-level model.

![CNN Transformer Structure](./fig/cnn_transformer_structure.png)

### LaBraM

See [Github](https://github.com/935963004/LaBraM) for details. 

In the project, we will only fine-tune the base model, for base model downloading, please refer to original repo. 

> We do not know how LaBraM works exactly, so the training configration is copied from original repo. If you have any suggestion (code of LaBraM model, training configration, etc), feel free to make a pull request or contact us.

### Gram

See [Github](https://github.com/iiieeeve/Gram) for details. 

The code structure of this model is inherited fro *LaBraM*. We only fine-tune it, too. Please refer to original repo for base model downloading.

> Have any suggestion on this model? Feel free to make a pull request!

### Qwen-VL

Instead of directly using NumPy arrays as input, some approaches convert EEG signals into images, transforming the task into a computer vision problem. Therefore, fine-tuning an existing vision model is a common approach, like [LLaVA-Med](https://github.com/microsoft/LLaVA-Med). Here, we will evaluate the ability of a base vision model to recognize EEG signals.

We will skip the training stage here. Since we will load the model locally, *make sure you have enough GPU memory!*

To accelerate evaluation, we can provide the model with a specific prompt that restricts the output to the class label, allowing for batch processing of samples.

```python
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="auto", trust_remote_code=True).eval()

image = ... # Assume a TUSE sample
prompt = "Based on the provided EEG, determine if the subject has seizure. If yes, answer 1; otherwise, answer 0. Any other responses are prohibited."

# Support batch input
query = tokenizer.from_list_format([
    {'image': image}, 
    {'text': prompt},
])
inputs = tokenizer(query, return_tensors='pt').to(model.device)
pred = model.generate(**inputs)

# Support batch decode
response = tokenizer.batch_decode(pred.cpu(), skip_special_tokens=True)
```

> More modes to be added...

## Getting Started

Follow thes  steps to set up the project locally

### Installation

```bash
git clone https://github.com/forLG/EEG-Baseline

conda create -n eeg-baseline
conda activate eeg-baseline

pip install -e .[dev]
```

### Data Setup

TODO: Release the dataset.

## Usage

This framework is operated through a single command line interface, `eeg-runner`, and configured with YAML files.

### Configuration

1. All experiment configurations are located in the `config/` directory.
2. Before running an experiment, copy an existing config(e.g, `configs/eegnet/tusz.yaml`), rename it, and change the configrations for your own experiment.
3. *Crucially, update 'path' related variable* in your new config to point to the correct location of your environment.

### Training a Model

To start a new training run, use the `train` command and provide the to your configuration file.

```bash
eeg-runner train --config configs/your_experiment_config.yaml
```

The training progress will be displayed in the console. By default:
- Logs will be written to a `.log` file in the `./logs/` directory.
- Model checkpoints will be saved in the `./checkpoints/` directory according to the strategy defined in your config.

### Evaluating a Model

To evaluate a previously trained model checkpoint, use the `evaluate` command.

```bash
eeg-runner evaluate --config configs/your_experiment_config.yaml --checkpoint ./checkpoints/your_training_checkpoint.pth
```

This will load the model weights from the checkpoint file and run an evaluation on the `eval` split of the dataset defined in the config.

*NOTES: The config in the evaluation stage is the same the training stage.*

## Extending the Framework

The framework is designed to be easily extensible.

### Adding a New Model

1. Create a new directory for your model: eeg_baseline/models/my_new_model/.

2. Inside this directory, create a `model.py` file containing your model's class definition (e.g., `MyNewModel`). If the model's original code has a non-standard interface, create a Wrapper class as the main entry point (see existing models for examples).

3. Add a new configuration file for your model's hyperparameters: `configs/model/my_new_model.yaml`.

4 You can now use `my_new_model` in your main experiment configuration file.

### Adding a New Dataset

1. If the dataset follows the same .npy + .json format:

    - No code changes are needed! Simply create a new experiment config file and set the data.name and data.path accordingly.

2. If the dataset has a new format:

    - Create a new dataset class in `eeg_baseline/datasets/` (e.g., `my_new_dataset.py`) that inherits from torch.utils.data.Dataset.

    - Update the factory function in `eeg_baseline/datasets/__init__.py` to recognize the new dataset name and use your new class.
