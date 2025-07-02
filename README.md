# 🌿 Pixel-LSTM Forecasting

Forecasting vegetation dynamics in Southern Africa using pixel-level LSTM models and remote sensing data.

This repository contains the code and data pipeline for predicting the Enhanced Vegetation Index (EVI) using a Long Short-Term Memory (LSTM) model. It integrates climate variables and static environmental features to model spatiotemporal vegetation change at 0.1° grid resolution.

---

## 🚀 Overview

- **Target Variable**: Enhanced Vegetation Index (EVI)
- **Model**: LSTM with static feature-based hidden state initialization
- **Region**: Southern Africa
- **Data Sources**: MODIS, ERA5, GMTED2010, GLDAS, SPEI
- **Forecast Horizon**: Multi-step sequence modeling
- **Goal**: Understand vegetation responses to hydroclimatic and geophysical variables

---

## 🧠 Features

- 📦 Raw NetCDF to training-ready `.npy` conversion
- 🧼 Temporal standardization & log-transformations
- 📍 Per-pixel sample selection & masking
- 🧠 LSTM model with custom static initialization
- 📉 Training with early stopping & loss tracking
- 🧪 Evaluation & result saving for downstream analysis

---

## 📁 Directory Structure

pixel-lstm-forecasting/
│
├── raw_data/ # Raw NetCDF files (ignored in .gitignore)
├── prepared_data/ # Processed NumPy arrays (X.npy, y.npy, etc.)
│
├── model_architecture.py # PixelLSTM model definition
├── dataloader_utils.py # Dataset, split logic, and loaders
├── prepare_data.py # Extracts X, y, static, coords from NetCDF
├── training.py # Main training pipeline
├── loading and using model.py # (Optional) inference or demo code
│
├── LICENSE
├── README.md





> 📎 Note: `raw_data/` and `prepared_data/` may be added via `.gitkeep` to persist empty folders.

---

## ⚙️ Installation

Make sure you have Python ≥3.8. Recommended to use a virtual environment.

```bash
pip install numpy pandas xarray torch scikit-learn matplotlib seaborn cartopy folium


## 📄 License

This project is licensed under the [MIT License](./LICENSE).
