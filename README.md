# Employee Attrition Prediction using Machine Learning

## 1. Problem Statement

Employee attrition is an important business problem because unexpected employee turnover can increase recruitment costs, reduce productivity, and affect team performance.

This project develops a machine learning based Employee Attrition Prediction System using the IBM HR Analytics Employee Attrition dataset.

The objective is to classify employees into two categories:

* **No** – Employee is predicted to stay
* **Yes** – Employee is predicted to leave

Five classification algorithms are implemented and evaluated using the same dataset and the same test split. An interactive Streamlit web application is also developed to allow users to upload the test dataset, select a classification model, and view its evaluation results.

---

## 2. Dataset Description

### Dataset

**IBM HR Analytics Employee Attrition & Performance Dataset**

The dataset contains information about employees, including demographic information, job characteristics, compensation, satisfaction levels, work experience, and other workplace-related attributes.

### Dataset Statistics

| Property            |                 Value |
| ------------------- | --------------------: |
| Total Instances     |                  1470 |
| Original Columns    |                    35 |
| Target Variable     |             Attrition |
| Features Used       |                    34 |
| Training Samples    |                  1176 |
| Testing Samples     |                   294 |
| Missing Values      |                     0 |
| Classification Type | Binary Classification |

The dataset satisfies the assignment requirements of:

* Minimum 500 instances
* Minimum 12 features

### Target Distribution

| Attrition | Count |
| --------- | ----: |
| No        |  1233 |
| Yes       |   237 |

The dataset is imbalanced because the number of employees who stayed is considerably higher than the number of employees who left.

### Preprocessing

The following preprocessing operations were performed:

1. Removed constant or identifier-related columns:

    * `EmployeeCount`
    * `EmployeeNumber`
    * `Over18`
    * `StandardHours`

2. Encoded the target variable:

    * `No → 0`
    * `Yes → 1`

3. Numerical features were standardized using `StandardScaler`.

4. Categorical features were converted using `OneHotEncoder`.

5. The dataset was divided using an 80:20 stratified train-test split.

6. The preprocessing object was fitted only on the training data and subsequently applied to the test data.

After preprocessing:

* Training data shape: **1176 × 51**
* Testing data shape: **294 × 51**

---

## 3. GitHub Repository

**GitHub Repository:**
`<TO-DO>`

The repository contains the complete source code, dataset used for testing, trained model files, preprocessing objects, evaluation results, README documentation, and Streamlit application.

---

## 4. Machine Learning Models

The following five classification models were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier

The models were evaluated using:

* Accuracy
* AUC Score
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

---

## 5. Model Comparison

| ML Model Name            | Accuracy |    AUC | Precision | Recall | F1 Score |    MCC |
| ------------------------ | -------: | -----: | --------: | -----: | -------: | -----: |
| Logistic Regression      |   0.8605 | 0.8115 |    0.6154 | 0.3404 |   0.4384 | 0.3871 |
| Decision Tree            |   0.7653 | 0.6105 |    0.3103 | 0.3830 |   0.3429 | 0.2036 |
| KNN                      |   0.8435 | 0.5946 |    0.5385 | 0.1489 |   0.2333 | 0.2222 |
| Naive Bayes              |   0.6463 | 0.7032 |    0.2605 | 0.6596 |   0.3735 | 0.2265 |
| Random Forest (Ensemble) |   0.8469 | 0.7856 |    0.6000 | 0.1277 |   0.2105 | 0.2254 |

---

## 6. Model Performance Observations

### Logistic Regression

Logistic Regression produced the strongest overall performance among the evaluated models. It achieved the highest accuracy, AUC, precision, F1 score, and MCC.

The model achieved an accuracy of 86.05% and an AUC of 0.8115. Its F1 score and MCC were also the highest among the models.

The results indicate that Logistic Regression provides a good balance between correctly identifying employees who stay and identifying employees who leave.

---

### Decision Tree

The Decision Tree achieved an accuracy of 76.53%, which was lower than the other major models.

Its recall for the attrition class was 0.3830, which was higher than Logistic Regression, KNN, and Random Forest, but its precision was only 0.3103.

The lower AUC, F1 score, and MCC indicate weaker overall predictive performance compared with Logistic Regression.

---

### K-Nearest Neighbors

KNN achieved an accuracy of 84.35%, which appears relatively high.

However, because the dataset is imbalanced, accuracy alone does not fully describe its performance. KNN obtained a low recall of 0.1489 for employees who actually left and an F1 score of 0.2333.

Its AUC of 0.5946 was also the lowest among the five models. Therefore, KNN was not considered the best model despite its relatively high accuracy.

---

### Naive Bayes

Naive Bayes achieved the highest recall for the attrition class at 0.6596.

This means it identified a larger proportion of employees who actually left compared with the other models.

However, its precision was only 0.2605, and its overall accuracy was 64.63%. Its MCC was also relatively low.

Therefore, Naive Bayes may be useful when identifying as many potential attrition cases as possible is the primary objective, but it does not provide the best overall balanced performance for this dataset.

---

### Random Forest

Random Forest achieved an accuracy of 84.69%, an AUC of 0.7856, and a precision of 0.6000.

Although its accuracy and precision were relatively strong, its recall for the attrition class was only 0.1277. Consequently, it missed a large proportion of employees who actually left.

Its F1 score of 0.2105 and MCC of 0.2254 were also lower than Logistic Regression.

Therefore, Random Forest did not outperform Logistic Regression on the overall evaluation criteria in this experiment.

---

## 7. Overall Winner

### Logistic Regression

**Logistic Regression was selected as the overall winner for this dataset.**

The main reasons are:

* Highest Accuracy: **0.8605**
* Highest AUC: **0.8115**
* Highest Precision: **0.6154**
* Highest F1 Score: **0.4384**
* Highest MCC: **0.3871**

Although Naive Bayes achieved the highest recall, Logistic Regression provided the best overall balance across the evaluation metrics.

The dataset is imbalanced, so model selection was not based on accuracy alone. AUC, precision, recall, F1 score, and MCC were also considered.

---

## 8. Streamlit Web Application

An interactive Streamlit application was developed to demonstrate the trained classification models.

### Application Features

The application provides:

1. **Test Dataset Upload**

    * Users can upload the `test_data.csv` file.
    * Only the test dataset is uploaded to keep the application lightweight.

2. **Model Selection**

    * Users can select one of the five implemented classification models from a dropdown.

3. **Evaluation Metrics**

    * Accuracy
    * AUC Score
    * Precision
    * Recall
    * F1 Score
    * MCC Score

4. **Classification Report**

    * Displays class-wise precision, recall, and F1 score.

5. **Confusion Matrix**

    * Provides a visual representation of classification results.

6. **Model Comparison**

    * Displays the performance of all implemented models for comparison.

### Live Streamlit Application

`<TO-DO>`

---

## 9. Project Structure

```text
employee-attrition-ml-app/
│
├── app.py
├── data_loader.py
├── preprocessing.py
├── train_models.py
├── evaluate_models.py
├── test_data.csv
├── requirements.txt
├── README.md
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── model/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── preprocessor.pkl
│   ├── label_encoder.pkl
│   └── model_comparison.csv
│
├── reports/
│
├── screenshots/
│
└── notebook/
```

---

## 10. Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit
* Joblib

---

## 11. Running the Project Locally

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Model Training

```bash
python train_models.py
```

### Run Model Evaluation

```bash
python evaluate_models.py
```

### Run Streamlit Application

```bash
streamlit run app.py
```

The application can then be accessed through the local Streamlit URL displayed in the terminal.

---

## 12. Academic Assignment Requirements Covered

| Requirement                | Status    |
| -------------------------- | --------- |
| Classification dataset     | Completed |
| Minimum 500 instances      | Completed |
| Minimum 12 features        | Completed |
| Logistic Regression        | Completed |
| Decision Tree              | Completed |
| KNN                        | Completed |
| Naive Bayes                | Completed |
| Random Forest              | Completed |
| Accuracy                   | Completed |
| AUC                        | Completed |
| Precision                  | Completed |
| Recall                     | Completed |
| F1 Score                   | Completed |
| MCC                        | Completed |
| Test dataset CSV           | Completed |
| Streamlit application      | Completed |
| Dataset upload             | Completed |
| Model selection dropdown   | Completed |
| Evaluation metrics display | Completed |
| Classification report      | Completed |
| Confusion matrix           | Completed |
| Model comparison           | Completed |

---

## 13. Conclusion

This project demonstrates an end-to-end machine learning workflow for employee attrition prediction, including dataset preprocessing, multiple classification algorithms, model evaluation, comparison, model persistence, and interactive deployment using Streamlit.

Among the evaluated models, Logistic Regression provided the strongest overall performance on the selected test dataset. The Streamlit application allows users to upload the test data, select different classification models, and inspect their evaluation results interactively.

````

### What we do next

Don't upload this to GitHub yet.

First, **open your `README.md` and replace everything with the above**. Keep the two placeholders:

```text
<ADD YOUR GITHUB REPOSITORY LINK HERE>
<ADD YOUR STREAMLIT COMMUNITY CLOUD LINK HERE>
````

We'll fill those **after deployment**.

Then we'll do one important final audit of your actual folder:

```text
app.py
data_loader.py
preprocessing.py
train_models.py
evaluate_models.py
requirements.txt
README.md
test_data.csv
model/
```

After that we'll check **`requirements.txt` → GitHub → Streamlit deployment → screenshot → final PDF**, so we don't miss another assignment requirement.