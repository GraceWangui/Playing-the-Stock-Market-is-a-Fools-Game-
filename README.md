# Playing-the-Stock-Market-is-a-Fools-Game-
Using Deep Learning to Predict Daily Stock Price Movements for 442 Companies


## **Overview**

This project explores the use of **deep learning** to predict stock price movements for **442 publicly traded companies**.
The dataset contains **daily percentage changes** in stock prices from **05 April 2010 to 31 March 2022**.
My primary task is to **predict the daily percentage movement for each company on 01 April 2022**—the first day following the historical data period.

This work integrates **Recurrent Neural Networks (RNNs)** for time series forecasting, **Optuna** for hyperparameter optimization, and **Captum** for model interpretability.

---

## **Project Objectives**

* **Forecasting Goal:** Predict daily percent change for each of the 442 companies on **01 April 2022**.
* **Performance Metric:** Mean Squared Error (MSE) between predictions and ground truth.
* **Optimization:** Use Optuna to fine-tune model hyperparameters for best performance.
* **Interpretability:** Apply Captum to understand which features influence predictions the most.

---

## **Dataset**

### **Training Data**

* **Period:** 05/04/2010 → 31/03/2022
* **Features:** Daily percent change in stock prices for 442 companies.
* **Target:** Percent change for each company on **01 April 2022**.

### **Submission Data**

* Contains IDs for prediction dates (01 April 2022).
* A **template submission file** with all 0.0 predictions is provided.

---

## **Approach**

### **1. Data Preparation**

* Load and preprocess stock price movement data.
* Handle missing values (if any) and ensure correct time ordering.
* Split data into **training**, **validation**, and **test** sets.
* Normalize/scale data for deep learning model training.

### **2. Model Architecture**
* Experiment with:

  * LSTM / GRU layers
  * Dropout for regularization
  * Sequence window size tuning
  * Fully connected output layer with 442 neurons (one per company)
* Loss function: **Mean Squared Error (MSE)**
* Optimizer: Adam / RMSprop

### **3. Hyperparameter Optimization**

* Use **Optuna** to:

  * Search optimal learning rate
  * Determine hidden layer size and depth
  * Tune dropout rate
  * Adjust sequence length
* Objective function: Validation MSE.

### **4. Prediction**

* Train final model on full training data with best hyperparameters.
* Predict **01 April 2022** stock movements.
* Save predictions in the **submission format**:

  ```csv
  ID,TARGET
  0,0.01234
  1,-0.00456
  ...
  ```

### **5. Model Interpretation**

* Use **Captum** to:

  * Identify important time steps and companies influencing predictions.
  * Generate attribution visualizations.
 * Provide analysis of observed trends.

---

## **Technologies Used**

* **Python 3**
* **PyTorch** (Deep Learning)
* **Optuna** (Hyperparameter Optimization)
* **Captum** (Model Interpretability)
* **Pandas, NumPy** (Data Handling)
* **Matplotlib / Seaborn** (Visualization)
* **Jupyter Notebook** (Experiment Tracking)

---

## **Folder Structure**

```
.
├── data/
│   ├── train.csv              # Historical stock movement data
│
├── notebooks/
│   ├── stock_prediction.ipynb # Main Jupyter notebook with analysis and training      
│
├── results/
│   ├── submission.csv         # Template for Final predictions
│   ├── captum_analysis/       # Feature attribution plots
│
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
```

---

## **How to Run**

### **1. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **2. Run Notebook**

Open the Jupyter notebook and execute all cells:

```bash
jupyter notebook notebooks/stock_prediction.ipynb
```

### **3. Generate Predictions**

* The notebook will output a `submission.csv` in `results/`.
* This file contains predicted percentage movements for all 442 companies.

---

## **Interpretability Insights**

Using **Captum**,:

* Identify which companies have the largest historical influence on predictions.
* Discover which days in the time window contribute most to forecasts.
* Analyze whether the model relies more on recent or long-term trends.

---

## **Disclaimer**

This project is a **technical demonstration** of applying deep learning to financial time series forecasting.
**It is not financial advice** and should not be used for real-world trading decisions.


