# ChurnIQ — Customer Churn Prediction

An interactive machine-learning web application that predicts customer churn, analyzes customer behavior, and helps businesses identify customers who may be at risk of leaving.

## 📌 Project Overview

Customer churn occurs when customers stop using a company's products or services. Identifying customers who are likely to leave helps businesses develop effective customer-retention strategies.

**ChurnIQ** is a web-based application built using Python, Streamlit, and machine-learning techniques. It analyzes customer information, predicts churn, visualizes customer trends, and presents model-performance metrics through an interactive dashboard.

## 🎯 Objectives

* Predict whether a customer is likely to leave a service.
* Analyze customer characteristics and churn patterns.
* Display customer statistics through interactive visualizations.
* Compare machine-learning models using cross-validation.
* Evaluate model performance using standard classification metrics.
* Provide predictions and retention recommendations through an intuitive interface.

## ✨ Key Features

### 1. Interactive Dashboard

Displays customer statistics and visualizations to help users understand the overall dataset and customer behavior.

### 2. Churn Analysis

Provides visual insights into customer churn patterns and supports analysis using available filters.

### 3. Customer Churn Prediction

Allows users to enter customer details and obtain a predicted churn classification and estimated churn probability.

### 4. Model Performance

Displays important evaluation metrics, including accuracy, precision, recall, F1-score, and ROC-AUC, along with a confusion matrix.

### 5. Machine-Learning Pipeline

Uses preprocessing and classification in a unified pipeline to prepare customer data and generate predictions.

## 🛠️ Technologies Used

| Technology   | Purpose                                            |
| ------------ | -------------------------------------------------- |
| Python       | Core programming language                          |
| Streamlit    | Interactive web application                        |
| Pandas       | Data loading and manipulation                      |
| Scikit-learn | Data preprocessing, model training, and evaluation |
| Joblib       | Saving and loading the trained model               |
| HTML and CSS | Custom dashboard styling, where applicable         |

## 🤖 Machine-Learning Approach

The project compares two classification algorithms:

**1. Logistic Regression**

A classification algorithm that estimates the probability of a customer belonging to the churn class.

**2. Random Forest Classifier**

An ensemble-learning algorithm that combines predictions from multiple decision trees.

### Model Selection

The training script compares the candidate models using stratified five-fold cross-validation with the F1-score as the selection metric.

The model with the better mean cross-validation F1-score is selected and saved as a trained preprocessing-and-classification pipeline.

The saved pipeline is subsequently loaded by the Streamlit application.

### Data Preprocessing

The preprocessing pipeline handles numerical and categorical features separately.

Depending on the feature type, preprocessing includes:

* Missing-value imputation.
* Standardization of numerical features.
* One-hot encoding of categorical features.

The target variable, `Churn`, represents whether a customer has left the service.

## 📊 Model Evaluation

The application currently displays the following evaluation results:

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 72.71% |
| Precision | 49.16% |
| Recall    | 78.61% |
| F1-score  |  0.605 |
| ROC-AUC   | 0.8361 |

**Note:** These are the values currently displayed by the application. They should be reported as final test-set results only after verifying the evaluation procedure and test split.

### Interpretation

* **Accuracy:** Measures the proportion of correct predictions.
* **Precision:** Measures how many customers predicted to churn actually churned.
* **Recall:** Measures how many actual churners were correctly identified.
* **F1-score:** Combines precision and recall into a single metric.
* **ROC-AUC:** Measures how well the model distinguishes churners from non-churners.

The results indicate that the model identifies a substantial proportion of customers who churn, while its precision leaves room for improvement.

## 📁 Project Structure

The project is organized as follows:

```text
CustomerChurnPrediction/
│
├── app.py
├── train_model.py
├── README.md
│
├── dataset/
│   └── customer_churn.csv
│
├── models/
│   └── churn_model.pkl
│
└── .venv/
```

**Note:** The structure above represents the main project files. Your repository may contain additional files.

## ⚙️ Installation and Setup

### Prerequisites

Install the following before running the project:

* Python
* pip
* Visual Studio Code or another Python editor

### Step 1: Open the Project Folder

Open the `CustomerChurnPrediction` folder in your terminal or Visual Studio Code.

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### Step 3: Install Dependencies

Install the packages required by the project:

```bash
pip install streamlit pandas scikit-learn joblib
```

If your application uses additional libraries, install those too.

### Step 4: Verify the Project Files

Make sure the following files are available:

* `app.py`
* `train_model.py`
* `dataset/customer_churn.csv`

If a compatible trained model already exists at `models/churn_model.pkl`, you can proceed to the next step.

### Step 5: Train the Model (If Required)

To train the model and generate the saved pipeline, run:

```bash
python train_model.py
```

Wait for training and model evaluation to finish.

This command should generate the trained model at:

```text
models/churn_model.pkl
```

Do not retrain the model unless required. Retraining may change the selected algorithm and evaluation results.

### Step 6: Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the local address displayed in the terminal to access ChurnIQ in your browser.

## 🔍 How to Use ChurnIQ

1. Launch the application.
2. Open the Dashboard to explore customer statistics.
3. Navigate to Churn Analysis to examine churn patterns.
4. Open Predict Churn and enter the required customer details.
5. View the predicted churn class, probability, and available recommendation.
6. Visit Model Performance to review the evaluation metrics.

## 🌍 Applications

Churn prediction can support businesses in:

* Identifying customers who may discontinue a service.
* Prioritizing customer-retention campaigns.
* Understanding customer churn patterns.
* Supporting data-driven retention decisions.
* Improving customer relationship management.

## ⚠️ Limitations

* Predictions depend on the quality and representativeness of the dataset.
* Model performance may change when applied to different customer populations.
* A predicted churn probability is an estimate, not a guarantee of customer behavior.
* Retention decisions should consider business context and the cost of false predictions.
* Model evaluation should use an appropriate held-out dataset to estimate performance on unseen customers.

## 🔮 Future Enhancements

Potential improvements include:

* Hyperparameter tuning and additional model comparisons.
* Improved precision while maintaining useful recall.
* Model explainability to identify factors influencing predictions.
* Integration with a live customer database or REST API.
* Deployment to a cloud-hosting platform.
* Monitoring prediction quality as new customer data becomes available.

## 👩‍💻 Conclusion

ChurnIQ demonstrates how machine learning and interactive data visualization can be combined to analyze customer behavior and predict potential customer churn.

The project provides an end-to-end workflow covering data preprocessing, model comparison, prediction, and performance evaluation through an interactive web application.

It serves as a practical demonstration of applying machine-learning techniques to a real-world business problem.
