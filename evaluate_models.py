import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

from preprocessing import preprocess_data


print("="*70)
print("MODEL EVALUATION")
print("="*70)


(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
    label_encoder,
    original_X_test

) = preprocess_data()



models = {

    "Logistic Regression":
        "model/logistic_regression.pkl",

    "Decision Tree":
        "model/decision_tree.pkl",

    "KNN":
        "model/knn.pkl",

    "Naive Bayes":
        "model/naive_bayes.pkl",

    "Random Forest":
        "model/random_forest.pkl"

}



os.makedirs(
    "reports",
    exist_ok=True
)



for name,path in models.items():


    print("\n")
    print("="*50)
    print(name)
    print("="*50)


    model = joblib.load(path)


    prediction = model.predict(
        X_test
    )


    print(
        classification_report(
            y_test,
            prediction,
            target_names=label_encoder.classes_
        )
    )



    cm = confusion_matrix(
        y_test,
        prediction
    )


    plt.figure(
        figsize=(5,4)
    )


    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )


    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.title(
        name
    )


    filename = (
            name.lower()
            .replace(" ","_")
            +"_confusion_matrix.png"
    )


    plt.savefig(
        f"reports/{filename}",
        bbox_inches="tight"
    )


    plt.close()



print("\nEvaluation completed successfully")