import os
import joblib
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Employee Attrition Intelligence",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

MODEL_DIRECTORY = "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl"
}


# ============================================================
# LOAD SAVED OBJECTS
# ============================================================

@st.cache_resource
def load_preprocessing_objects():

    preprocessor = joblib.load(
        os.path.join(
            MODEL_DIRECTORY,
            "preprocessor.pkl"
        )
    )

    label_encoder = joblib.load(
        os.path.join(
            MODEL_DIRECTORY,
            "label_encoder.pkl"
        )
    )

    return preprocessor, label_encoder


@st.cache_resource
def load_model(model_file):

    return joblib.load(
        os.path.join(
            MODEL_DIRECTORY,
            model_file
        )
    )


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("📊 Employee Attrition Intelligence System")

st.markdown(
    """
This application demonstrates machine learning models for
**employee attrition prediction** using the IBM HR Analytics dataset.

Upload the **test dataset**, select a classification model,
and evaluate its performance using the required assignment metrics.
"""
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Model Evaluation")

st.sidebar.markdown(
    """
### Available Models

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- Naive Bayes
- Random Forest
"""
)

selected_model_name = st.sidebar.selectbox(
    "Select Classification Model",
    list(MODEL_FILES.keys())
)


# ============================================================
# TEST DATA UPLOAD
# ============================================================

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload the CSV test data used for evaluation",
    type=["csv"]
)


if uploaded_file is None:

    st.info(
        "Please upload the test_data.csv file to begin model evaluation."
    )

    st.stop()


# ============================================================
# READ TEST DATA
# ============================================================

try:

    test_data = pd.read_csv(
        uploaded_file
    )

except Exception as error:

    st.error(
        f"Unable to read the uploaded CSV file: {error}"
    )

    st.stop()


# ============================================================
# VALIDATE TEST DATA
# ============================================================

required_target = "Attrition"

if required_target not in test_data.columns:

    st.error(
        "The uploaded CSV must contain the 'Attrition' column "
        "because evaluation requires the actual target values."
    )

    st.stop()


# ============================================================
# DISPLAY DATASET
# ============================================================

st.header("2. Test Dataset")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Test Instances",
        len(test_data)
    )

with col2:

    st.metric(
        "Columns",
        len(test_data.columns)
    )

with col3:

    st.metric(
        "Attrition Cases",
        int(
            (test_data["Attrition"] == "Yes").sum()
        )
    )


with st.expander("Preview Uploaded Data"):

    st.dataframe(
        test_data.head(10),
        use_container_width=True
    )


# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================

X_test_raw = test_data.drop(
    columns=["Attrition"]
)

y_test_text = test_data["Attrition"]


try:

    preprocessor, label_encoder = load_preprocessing_objects()

    y_test = label_encoder.transform(
        y_test_text
    )

except Exception as error:

    st.error(
        f"Unable to prepare the uploaded data: {error}"
    )

    st.stop()


# ============================================================
# TRANSFORM TEST DATA
# ============================================================

try:

    X_test_processed = preprocessor.transform(
        X_test_raw
    )

except Exception as error:

    st.error(
        f"The uploaded dataset does not match the expected "
        f"feature structure: {error}"
    )

    st.stop()


# ============================================================
# LOAD SELECTED MODEL
# ============================================================

try:

    selected_model = load_model(
        MODEL_FILES[selected_model_name]
    )

except Exception as error:

    st.error(
        f"Unable to load the selected model: {error}"
    )

    st.stop()


# ============================================================
# MODEL PREDICTION
# ============================================================

try:

    y_pred = selected_model.predict(
        X_test_processed
    )

    y_probability = selected_model.predict_proba(
        X_test_processed
    )[:, 1]

except Exception as error:

    st.error(
        f"Prediction failed: {error}"
    )

    st.stop()


# ============================================================
# CALCULATE REQUIRED METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

auc_score = roc_auc_score(
    y_test,
    y_probability
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

mcc = matthews_corrcoef(
    y_test,
    y_pred
)


# ============================================================
# DISPLAY SELECTED MODEL
# ============================================================

st.header("3. Model Evaluation")

st.success(
    f"Currently evaluating: {selected_model_name}"
)


# ============================================================
# METRICS
# ============================================================

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:

    st.metric(
        "Accuracy",
        f"{accuracy:.4f}"
    )

    st.metric(
        "Precision",
        f"{precision:.4f}"
    )


with metric_col2:

    st.metric(
        "AUC Score",
        f"{auc_score:.4f}"
    )

    st.metric(
        "Recall",
        f"{recall:.4f}"
    )


with metric_col3:

    st.metric(
        "F1 Score",
        f"{f1:.4f}"
    )

    st.metric(
        "MCC Score",
        f"{mcc:.4f}"
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

st.header("4. Classification Report")

report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report
).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.header("5. Confusion Matrix")

matrix = confusion_matrix(
    y_test,
    y_pred
)

fig, ax = plt.subplots(
    figsize=(6, 4)
)

sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
    ax=ax
)

ax.set_xlabel(
    "Predicted Label"
)

ax.set_ylabel(
    "Actual Label"
)

ax.set_title(
    f"{selected_model_name} - Confusion Matrix"
)

st.pyplot(
    fig
)

plt.close(fig)


# ============================================================
# MODEL COMPARISON
# ============================================================

st.header("6. Model Comparison")

comparison_file = os.path.join(
    MODEL_DIRECTORY,
    "model_comparison.csv"
)

if os.path.exists(comparison_file):

    comparison_df = pd.read_csv(
        comparison_file
    )

    st.dataframe(
        comparison_df.round(4),
        use_container_width=True
    )

    st.caption(
        "The comparison table shows the results obtained during "
        "the original model evaluation on the project test split."
    )

else:

    st.warning(
        "model_comparison.csv was not found."
    )


# ============================================================
# INTERPRETATION
# ============================================================

st.header("7. Model Interpretation")

if selected_model_name == "Logistic Regression":

    st.write(
        """
        Logistic Regression provides a strong overall balance between
        discrimination and classification performance for this dataset.
        Its AUC and MCC values make it a useful baseline for employee
        attrition prediction.
        """
    )

elif selected_model_name == "Decision Tree":

    st.write(
        """
        The Decision Tree provides an interpretable rule-based model.
        However, its overall performance is lower than Logistic Regression
        on the current test split.
        """
    )

elif selected_model_name == "K-Nearest Neighbors":

    st.write(
        """
        KNN achieves reasonable accuracy but has relatively low recall
        for employees who actually leave, which limits its usefulness
        when identifying potential attrition cases.
        """
    )

elif selected_model_name == "Naive Bayes":

    st.write(
        """
        Naive Bayes identifies a larger proportion of actual attrition
        cases, resulting in higher recall, but it also produces more
        false positives and therefore lower precision.
        """
    )

elif selected_model_name == "Random Forest":

    st.write(
        """
        Random Forest provides strong accuracy and precision, but its
        recall for attrition cases is relatively low on the current
        test split.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "IBM HR Analytics Employee Attrition | "
    "Classification Model Evaluation"
)