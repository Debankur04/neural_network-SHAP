# 📁 streamlit/

This folder contains the **Streamlit dashboard** — an interactive web app that lets anyone use the trained model and explore SHAP explanations without writing code.

## What belongs here

| File | Purpose |
|---|---|
| `app.py` | Main entry point — run this to launch the dashboard |
| `pages/` | Additional pages if the app grows multi-page |
| `components/` | Reusable UI components (charts, input forms, etc.) |
| `assets/` | Static files like logos, custom CSS |

## How to run the app

Make sure your virtual environment is active and the model is trained first (see `model/README.md`), then:

```bash
streamlit run streamlit/app.py
```

Open your browser at `http://localhost:8501`.

## What the app should do (planned features)

- [ ] Upload a CSV dataset or use a built-in sample
- [ ] Run the trained neural network on the input
- [ ] Display prediction results clearly
- [ ] Show SHAP summary plot (global feature importance)
- [ ] Show SHAP waterfall plot for a single prediction (local explanation)
- [ ] Allow users to tweak input values and see how predictions change

## Rules for contributors

- Keep `app.py` as the **entry point only** — actual logic should be imported from `model/` or helper files inside `streamlit/`.
- Do not duplicate model or SHAP code here. Import from `model/` instead.
- Every UI element should have a label or tooltip explaining what it does — remember, end users may not be ML engineers.
- Test the app locally before submitting a PR. Include a screenshot in your PR if you changed any UI.
- Avoid hardcoding file paths. Read them from `config/config.yaml`.

## Streamlit tips for new contributors

- `st.write()` prints almost anything to the page — great for quick debugging
- `st.sidebar` puts controls on the left panel
- `st.pyplot()` renders matplotlib/SHAP plots directly
- Hot reload is on by default — save your file and the browser refreshes automatically
- Docs → [docs.streamlit.io](https://docs.streamlit.io)
