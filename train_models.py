import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from preprocessing import preprocess_data


print("\n")
print("=" * 70)
print("MODEL TRAINING")
print("=" * 70)


# --------------------------------------------------
# Load preprocessed data
# --------------------------------------------------

(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
    label_encoder,
    original_X_test
) = preprocess_data()



# --------------------------------------------------
# Create model directory
# --------------------------------------------------

os.makedirs("model", exist_ok=True)



# --------------------------------------------------
# Define Models
# --------------------------------------------------

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            random_state=42
        ),


    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),


    "KNN":
        KNeighborsClassifier(
            n_neighbors=5
        ),


    "Naive Bayes":
        GaussianNB(),


    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
}



results = []



# --------------------------------------------------
# Train and Evaluate Models
# --------------------------------------------------

for model_name, model in models.items():

    print("\n")
    print("=" * 70)
    print(model_name)
    print("=" * 70)


    # Training

    model.fit(
        X_train,
        y_train
    )


    # Prediction

    y_pred = model.predict(
        X_test
    )


    y_prob = model.predict_proba(
        X_test
    )[:,1]



    # Metrics

    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    auc = roc_auc_score(
        y_test,
        y_prob
    )


    precision = precision_score(
        y_test,
        y_pred
    )


    recall = recall_score(
        y_test,
        y_pred
    )


    f1 = f1_score(
        y_test,
        y_pred
    )


    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )



    print(f"Accuracy  : {accuracy:.4f}")
    print(f"AUC       : {auc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"MCC       : {mcc:.4f}")



    results.append({

        "Model": model_name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc

    })



    # Save trained model

    filename = (
            model_name
            .lower()
            .replace(" ", "_")
            + ".pkl"
    )


    joblib.dump(
        model,
        f"model/{filename}"
    )



# --------------------------------------------------
# Save preprocessing objects
# --------------------------------------------------

joblib.dump(
    preprocessor,
    "model/preprocessor.pkl"
)


joblib.dump(
    label_encoder,
    "model/label_encoder.pkl"
)



# --------------------------------------------------
# Results Table
# --------------------------------------------------

results_df = pd.DataFrame(results)


print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)


print(
    results_df.round(4)
)



# Save comparison

results_df.to_csv(
    "model/model_comparison.csv",
    index=False
)



# --------------------------------------------------
# Save test data for Streamlit
# --------------------------------------------------

test_data = original_X_test.copy()

test_data["Attrition"] = label_encoder.inverse_transform(
    y_test
)


test_data.to_csv(
    "test_data.csv",
    index=False
)


print("\n")
print("=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)