from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from data_loader import df


def preprocess_data():

    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)


    # Remove unnecessary columns

    columns_to_drop = [
        "EmployeeCount",
        "EmployeeNumber",
        "Over18",
        "StandardHours"
    ]

    data = df.drop(columns=columns_to_drop)


    print("\nDropped Columns:")
    print(columns_to_drop)


    # Split features and target

    X = data.drop("Attrition", axis=1)
    y = data["Attrition"]


    # Encode target

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(y)


    print("\nTarget Encoding:")
    print(
        dict(
            zip(
                label_encoder.classes_,
                label_encoder.transform(label_encoder.classes_)
            )
        )
    )


    # Identify feature types

    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()


    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()


    print("\nNumeric Features :", len(numeric_features))
    print("Categorical Features :", len(categorical_features))


    print("\nCategorical Columns:")
    print(categorical_features)



    # Create preprocessing pipeline

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_features
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )


    # Train test split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


    print("\nTrain/Test Split")

    print(f"Training Samples : {len(X_train)}")
    print(f"Testing Samples  : {len(X_test)}")


    # Fit only on training data

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )


    print("\nProcessed Data Shapes")

    print("X_train :", X_train_processed.shape)
    print("X_test  :", X_test_processed.shape)


    print("\nPreprocessing Completed Successfully")


    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor,
        label_encoder,
        X_test
    )



if __name__ == "__main__":

    preprocess_data()