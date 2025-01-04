# ML-Based Stroke Risk Prediction

## Project Overview

The primary objective of this project is to develop a predictive model that can accurately determine the presence of heart disease in patients based on a set of medical attributes. This model aims to assist in early diagnosis and potential intervention, improving patient outcomes and healthcare efficiency.

The project involves analyzing the Heart Disease dataset from the UCI Machine Learning Repository using Python and Jupyter Notebook. The analysis will leverage libraries such as numpy and pandas for data manipulation, and sklearn.model_selection for splitting the dataset into training and test sets. Logistic Regression will be used to build a predictive model to determine the presence of heart disease based on various medical attributes.

## Dataset

The dataset used for this project is the [Heart Disease dataset from the UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease).

The dataset contains various features such as age, sex, blood pressure, cholesterol levels, and other medical attributes. These features are used to predict whether a patient has heart disease or not.

## Installation

To run this project on your local machine, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/ML-Based-Stroke-Risk-Prediction.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

- `data/`: Folder containing the dataset.
- `assets/`: Folder containing saved models and output images.
- `README.md`: This file with an overview of the project.

## Workflow

1. **Data Preprocessing**:

   - Importing and cleaning the dataset.
   - Handling missing values and encoding categorical variables.
   - Normalizing/standardizing features for model training.

2. **Model Building**:

   - Splitting the dataset into training and test sets using `sklearn.model_selection`.
   - Training the Logistic Regression model on the training data.
   - Evaluating the model's performance using accuracy and other metrics.

3. **Results and Visualizations**:

   - Visualizing the model's performance through various graphs and plots.
   - Output images showing the confusion matrix, accuracy, and other evaluation metrics.

4. **Deployment**:
   - The model can be integrated into web applications for real-time predictions.

## Output Images

In the `assets/` folder, you can find images such as:

## References

- UCI Heart Disease Dataset: [Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- Logistic Regression in Scikit-Learn: [Sklearn Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- Pandas Documentation: [Pandas Docs](https://pandas.pydata.org/pandas-docs/stable/)
- Numpy Documentation: [Numpy Docs](https://numpy.org/doc/stable/)
