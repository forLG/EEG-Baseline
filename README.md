# EEG Classification Baselines

The goal of this project is to establish a set of reproducible baseline results for EEG signal classification.

## Dataset

This project is designed to work with the following EEG datasets.

| Dataset | Train | Eval | Labels |
|---------|-------|------|--------|
| *HMC* | first 100 subjects | last 26 subjects | ? |
| *TUSE* | train | eval | seizure (or not) |

<!-- Dataset link -->

Before starting, we may need to preprocess the dataset.

<!-- Dataset preprocessing -->

## Models

### EEGNet

A compact convolutional neural network for EEG-based brain-computer interfaces.

Set the parameters as in the [Paper](https://arxiv.org/pdf/1611.08024).

![EEGNet Structure](./fig/eegnet_structure.png)

### CNN+Transformers

A CNN+Transformer architecture is a common baseline that appears in many papers on EEG models, but a canonical implementation is not readily available. Therefore, we will implement, train, and evaluate a simple version here. The basic structure is outlined below.

```python
class PositionEncoding(nn.Module):
    ...

class CNNTransformers(nn.Module):
    def __init__(...):
        # CNN feature extractor
        self.cnn_extractor = nn.Sequential(
            ...
        )
        # Add position information
        self.pos_encoder = PositionEncoding(...)
        # Transformer 
        self.transformer_encoder = nn.TransformerEncoder(...)
        # Classification head
        self.classification_head = nn.Sequential(
            ...
        )
    ...
```

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
