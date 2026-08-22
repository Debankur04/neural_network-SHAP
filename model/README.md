# 📁 model/

This folder contains the **core neural network code** — model architecture, training logic, evaluation, and SHAP integration. This is the heart of the project.

## What belongs here

| File | Purpose |
|---|---|
| `architecture.py` | Neural network class definition (layers, forward pass) |
| `train.py` | Training loop — loads data, trains the model, saves weights |
| `evaluate.py` | Runs the model on test data and reports metrics |
| `shap_explain.py` | Computes SHAP values and generates explanation outputs |
| `saved_weights/` | Folder where trained model checkpoints are saved |

## How training works (overview)

1. `train.py` reads settings from `config/config.yaml`
2. It loads the dataset, builds the model from `architecture.py`, and runs the training loop
3. After training, the model weights are saved to `saved_weights/`
4. `shap_explain.py` loads those weights and computes SHAP values on a test set

## Rules for contributors

- **Model architecture changes** should come with a comment explaining *why* — what problem does this change solve?
- Keep `train.py` clean and readable. Complex logic should live in helper functions with clear names.
- `saved_weights/` should be in `.gitignore` — model files are too large for Git. Use a release or external storage instead.
- If you change the model output shape or the expected input format, update `shap_explain.py` and the Streamlit app too.
- Write docstrings for every class and function. Other contributors need to understand your code without running it first.

## Example docstring format

```python
def forward(self, x):
    """
    Forward pass through the network.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, input_features)

    Returns:
        torch.Tensor: Output predictions of shape (batch_size, 1)
    """
```
