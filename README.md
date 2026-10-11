# Fruit & Vegetable Classifier — Streamlit

This app uses the CNN from `Fruits_Vegetables.ipynb`. Your notebook defines an input size of **128 × 128 × 3** and a 36-unit softmax output. The app expects the saved model and labels from that same training run.

## 1. Save your trained model and labels in Colab

Run this **after `model.fit(...)`** (and after `train_data` has been created). It saves both files into the folder where the notebook is running:

```python
import json

model.save("model.keras")

with open("class_names.json", "w", encoding="utf-8") as f:
    json.dump(train_data.class_names, f, ensure_ascii=False, indent=2)

print("Saved model.keras and class_names.json")
```

Download `model.keras` and `class_names.json` from Colab to your computer. Put both files in the same folder as `app.py`.

> Important: save labels from `train_data.class_names` in the exact training run used for this model. Do not type the 36 class names manually or sort them separately.

## 2. Folder structure

Keep these files together:

```text
fruits_vegetables_streamlit/
├── app.py
├── requirements.txt
├── README.md
├── model.keras          # you add this from Colab
└── class_names.json     # you add this from Colab
```

## 3. Run on your computer

Install Python 3.10 or 3.11 for best compatibility, then open a terminal in this folder and run:

```bash
python -m venv .venv
```

Activate it:

**Windows (Command Prompt):**
```bat
.venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

Install packages and start Streamlit:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Your browser should open the local app. If it does not, use the local URL shown in the terminal (usually `http://localhost:8501`).

## 4. Deploy online with Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, `class_names.json`, and `model.keras`.
3. Go to https://share.streamlit.io/ and connect the repository.
4. Choose `app.py` as the app's main file and deploy.

**Model file note:** GitHub has file-size limits. If `model.keras` is too large, use Git LFS or another supported model-hosting method and adjust the app to load it from there. Never upload private data or credentials.

## Notes

- The app resizes uploaded pictures to 128×128 and converts them to RGB.
- Your notebook's model has a 36-class output. The saved class-name list must contain exactly 36 labels in the correct order.
- The displayed confidence is not a guarantee. A model trained for a fixed set of classes will still choose among those classes for an unfamiliar object; it does not automatically know every new fruit or vegetable.
