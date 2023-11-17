# Customer Churn Prediction README

This repository contains code for predicting customer churn using machine learning techniques. The code is organized into several sections, each serving a specific purpose in the data preprocessing, exploratory data analysis (EDA), and model building process.

`note.ipynb` contains code for training the model and `deploy.py` contain code for deploying the model. A video is also included showing how the deployed web app works.

## Overview of Contents

1. **Data Preprocessing:**
    - In this section, the code loads the dataset (`CustomerChurn_dataset.csv`) and performs necessary preprocessing steps, such as handling missing values, encoding categorical variables, and scaling numeric features.

2. **Exploratory Data Analysis (EDA):**
    - The EDA section explores the dataset to gain insights into the distribution and relationships between different features. It includes visualizations to analyze class imbalance, data correlations, and the impact of various features on customer churn.

3. **Feature Selection:**
    - The code uses a Random Forest classifier to identify the top 10 most important features for predicting customer churn. These features are selected for further model training.

4. **SMOTE (Synthetic Minority Over-sampling Technique):**
    - As the dataset exhibits class imbalance in the target variable (Churn), the code applies SMOTE to oversample the minority class and balance the dataset.

5. **MLP Model Training:**
    - The repository includes code to train a Multi-Layer Perceptron (MLP) model using Keras with TensorFlow backend. The model is trained on the oversampled data with hyperparameter tuning using GridSearchCV.

6. **Model Evaluation:**
    - The trained model is evaluated on both the oversampled dataset and the original unbalanced dataset. The evaluation includes accuracy and Area Under the Curve (AUC) score calculations.

7. **Optimizing the Model:**
    - The code further explores model optimization by checking for multicollinearity in the selected features using Variance Inflation Factor (VIF).

## How to Use

1. **Environment Setup:**
    - Ensure you have the necessary Python libraries installed. You can install them using the requirements file:
      ```
      pip install -r tensorflow, keras, scikit-learn
      ```

2. **Data:**
    - Place your dataset (CSV format) in the root directory with the name `CustomerChurn_dataset.csv`.

3. **Run the Code:**
    - Execute the code in a Jupyter Notebook or script, following the order of sections outlined in the code file (`customer_churn_prediction.ipynb`).

4. **Results and Visualizations:**
    - Examine the visualizations and results generated during the EDA to gain insights into customer churn factors.

5. **Model Training and Evaluation:**
    - Review the MLP model training, evaluation, and optimization steps. Experiment with different hyperparameters if necessary.

6. **Notebook Organization:**
    - The code is organized into sections and comments for clarity. Follow the flow of the notebook for a step-by-step understanding of the analysis and model building.

Feel free to modify and adapt the code based on your specific dataset and requirements. If you encounter any issues or have questions, please refer to the documentation or reach out to the repository owner.

Enjoy exploring and predicting customer churn!
