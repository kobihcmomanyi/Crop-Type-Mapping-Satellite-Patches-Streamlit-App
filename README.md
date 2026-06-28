# Crop Type Mapping from Satellite Image Patches

A deep learning project that classifies satellite image patches (EuroSAT RGB) into crop/land-use types using and comparing four CNN architectures, deployed as an interactive Streamlit web app for live predictions on uploaded images.

**Live App:** [https://crop-type-mapping-satellite-patches.streamlit.app](https://crop-type-mapping-satellite-patches.streamlit.app)

---

## Overview

This project trains and benchmarks four convolutional neural networks on the [EuroSAT](https://madm.dfki.de/files/sentinel/EuroSAT.zip) RGB dataset, restricted to three classes:

- **AnnualCrop**
- **PermanentCrop**
- **Pasture**

The best-performing model (selected by test accuracy) is deployed in a Streamlit app that lets users upload a satellite image patch and get a predicted class with confidence scores.

## Models Trained & Compared

| Model | Type |
|---|---|
| Custom CNN | Built from scratch |
| MobileNetV2 | Transfer learning (ImageNet weights) |
| ResNet50 | Transfer learning (ImageNet weights) |
| EfficientNetB0 | Transfer learning (ImageNet weights) |

Each model is evaluated on accuracy, precision, recall, F1, ROC-AUC, and inference time. The highest-accuracy model is automatically selected as `best_model` and used for confusion matrix analysis, ROC/PR curves, and Grad-CAM explainability.

## Pipeline Summary

1. **Data acquisition** — Download and unzip the EuroSAT RGB dataset.
2. **Data cleaning** — Keep only the 3 target classes; remove corrupted, duplicate, and blurry images.
3. **Preprocessing** — Resize to 224×224, rescale pixel values to [0, 1], with augmentation (rotation, zoom, flips, brightness, channel shift) on the training set.
4. **Train/val/test split** — Stratified 70/15/15 split.
5. **Training** — All four models trained with early stopping and learning rate reduction on plateau.
6. **Evaluation** — Classification reports, confusion matrix, ROC/PR curves, Grad-CAM visualizations.
7. **Deployment** — Best model served via a Streamlit app for interactive predictions.

## Repository Structure

```
.
+-- app.py                                                  # Streamlit app
+-- crop_classifier.keras                                   # Saved best-performing model
+-- class_indices.json                                      # Index-to-class-label mapping
+-- requirements.txt                                        # Python dependencies
+-- README.md
```

## Running the App Locally

```bash
git clone https://github.com/kobihcmomanyi/crop-type-mapping-satellite-patches.git
cd crop-type-mapping-satellite-patches
pip install -r requirements.txt
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

### Using `tensorflow-cpu` instead of `tensorflow`

`requirements.txt` specifies `tensorflow-cpu` rather than the full `tensorflow` package. Since this app only runs inference (no GPU training), `tensorflow-cpu` is a lighter-weight, CPU-only build that:

- Installs faster and takes up less disk space
- Speeds up the build on Streamlit Community Cloud (which runs on CPU anyway)
- Avoids pulling in CUDA/GPU dependencies that aren't needed for serving predictions

If you have a GPU locally and want to use it for local testing, you can swap `tensorflow-cpu` for `tensorflow` in `requirements.txt` — the app code itself doesn't need to change either way.

### Troubleshooting: NumPy/Pandas binary incompatibility error

If you see this error when running `streamlit run app.py`:

```
ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
```

This happens when installed NumPy and Pandas versions were built against incompatible binary ABIs (common after installing packages separately or upgrading one but not the other). Fix it by upgrading both to their latest versions:

```bash
pip install --upgrade numpy pandas
```

Then re-run `streamlit run app.py`.

## Using the Live App

1. Open [https://crop-type-mapping-satellite-patches.streamlit.app](https://crop-type-mapping-satellite-patches.streamlit.app).
2. Upload a satellite image patch (JPG, PNG, or TIFF) — ideally a EuroSAT-style RGB patch (~64×64 to a few hundred pixels per side; the app resizes automatically).
3. The app displays:
   - The uploaded image
   - The predicted crop/land type (`AnnualCrop`, `PermanentCrop`, or `Pasture`)
   - The model's confidence score
   - A bar chart of class probabilities

## Deployment Details (Streamlit Community Cloud)

The app is deployed on [Streamlit Community Cloud](https://streamlit.io/cloud), which builds directly from this GitHub repository.

**Deployment steps used:**

1. Trained model saved to disk in `.keras` format, along with a `class_indices.json` mapping of class indices to label names.
2. Repository pushed to GitHub, including `app.py`, the saved model, the class index mapping, and `requirements.txt`.
   - If the model file exceeds GitHub's 100 MB file limit, it is tracked via [Git LFS](https://git-lfs.github.com/) or hosted externally (e.g. Hugging Face Hub) and downloaded at app startup.
3. On [share.streamlit.io](https://share.streamlit.io), a new app was created by selecting this repository, the `main` branch, and `app.py` as the entry point.
4. Streamlit Cloud installs dependencies from `requirements.txt` and launches the app automatically on every push to `main`.
5. The app is served at the public URL:
   **[https://crop-type-mapping-satellite-patches.streamlit.app](https://crop-type-mapping-satellite-patches.streamlit.app)**

**Key implementation notes:**
- Inference preprocessing in `app.py` mirrors training preprocessing exactly: resize to 224×224 and rescale pixel values by `1/255`.
- The model is loaded once and cached via `@st.cache_resource` to avoid reloading on every user interaction.
- Predicted class indices are mapped back to human-readable labels using `class_indices.json`, generated from the training data generator's `class_indices` attribute.

## Tech Stack

- **Modeling:** TensorFlow / Keras
- **Data handling:** Pandas, NumPy, OpenCV, Pillow
- **Evaluation:** scikit-learn (classification metrics, ROC/PR curves)
- **Explainability:** Grad-CAM
- **Deployment:** Streamlit, Streamlit Community Cloud

## Dataset

[EuroSAT](https://madm.dfki.de/files/sentinel/EuroSAT.zip) — Sentinel-2 satellite image patches covering 10 land use/cover classes; this project uses the RGB version restricted to `AnnualCrop`, `PermanentCrop`, and `Pasture`.
