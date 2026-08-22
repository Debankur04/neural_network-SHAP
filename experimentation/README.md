# 📁 experimentation/

This folder is your **sandbox**. It is where ideas are tested, datasets are explored, and SHAP experiments are run before anything gets moved into production code.

## What belongs here

| Type | Examples |
|---|---|
| Jupyter Notebooks | `01_data_exploration.ipynb`, `02_shap_analysis.ipynb` |
| Standalone scripts | `test_shap_values.py`, `compare_models.py` |
| Output plots | SHAP summary plots, feature importance charts |
| Temporary data samples | Small CSVs used only for testing |

## Naming convention

Please name your files with a number prefix so they stay in logical order:

```
01_data_exploration.ipynb
02_baseline_model.ipynb
03_shap_deep_dive.ipynb
```

This makes it easy for anyone new to follow the progression of experiments.

## Rules for contributors

- This folder is **exploratory** — messy work is okay here, but add a comment at the top of each file explaining what it is trying to do.
- Do **not** put finalized model code here. Once something is working and clean, move it to `model/`.
- Large datasets should **not** be committed. Use `.gitignore` or store them externally and reference the path in `config/config.yaml`.
- If your notebook produces important results or plots, mention it in your Pull Request description so reviewers know what to look at.

## How to run notebooks

Make sure your virtual environment is active (see main README), then:

```bash
jupyter notebook
```

This opens a browser tab where you can navigate to and run any `.ipynb` file in this folder.
