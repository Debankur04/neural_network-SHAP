# 📁 config/

This folder holds all **configuration files** for the project — things like model hyperparameters, dataset paths, and training settings.

## What belongs here

| File | Purpose |
|---|---|
| `config.yaml` | Main configuration file (hyperparameters, paths, settings) |
| `config.example.yaml` | A template showing all available config keys — copy this to get started |

## How to use

1. Copy the example config:
   ```bash
   cp config.example.yaml config.yaml
   ```
2. Edit `config.yaml` with your own values (dataset path, learning rate, epochs, etc.)
3. The training and experiment scripts will read from `config.yaml` automatically

## Rules for contributors

- **Never commit `config.yaml`** — it may contain local paths or sensitive values. It is listed in `.gitignore`.
- **Always update `config.example.yaml`** when you add a new config key. This is how other contributors know what options exist.
- Keep keys grouped logically (e.g., all model settings together, all data settings together).
- Add a comment above every key explaining what it does.

## Example structure for `config.yaml`

```yaml
# Model settings
model:
  hidden_layers: 3
  neurons_per_layer: 128
  activation: relu
  dropout: 0.2

# Training settings
training:
  epochs: 50
  learning_rate: 0.001
  batch_size: 32
  optimizer: adam

# Data settings
data:
  train_path: data/train.csv
  test_path: data/test.csv
  target_column: label
```
