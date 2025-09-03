# 🌱 Predicting Food Waste By Multiple Linear Regression
---

## Table of Contents
- [Overview](#-overview)
- [Getting started (Jupyter Notebook)](#getting-started-jupyter-notebook)
- [Description](#descripton)
- [Features](#features)
- [Command Examples](#)
- [Reference](#reference)
- [Contribution](#contribution)
---

## 📝 Overview

A Streamlit-powered web application designed to predict and visualize future food waste quantities in restaurant and event settings.
Users input key operational variables—such as food type, preparation method, event type, and guest count—and receive real-time predictions powered by a Multiple Linear Regression (MLR) model. 
The tool is intended to support data-driven decision-making for event planners, chefs, and sustainability officers aiming to reduce overproduction and improve planning accuracy.
---

## Key Features

- `Future Waste Prediction` using multiple operational indicators
- `Interactive Input Form` for food type, event type, number of guests, and more
- `Regression Model Evaluation` with metrics such as MAE, RMSE, and MAPE
- `Visual Analytics Dashboard` with bar charts, scatter plots, and heatmaps
- `Streamlit Interface` enabling real-time simulation and planning support
- `Modular Python Codebase` built with pandas, scikit-learn, and matplotlib
---

## Getting started (Jupyter Notebook)
1. Create conda virtual environment
``` bash
conda create -n ddw_dtp python=3.12 anaconda mypy nb_mypy
```

2. Activate your ddw_dtp:
``` bash
conda activate ddw_dtp
```
---

## Getting started (Streamlit WebApp)
1.
``` bash
pip install pipenv
```

2. 
``` bash
python -m pipenv install
```

3. 
``` bash
python -m pipenv shell
```
or 
``` bash
python -m pipenv run
```

2. 
``` bash
streamlit run Home.py
```
---

