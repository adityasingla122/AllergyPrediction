import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import time

# --- Configuration ---
CSV_FILE_PATH = 'merged_data.csv'
TARGET_LB_COLUMN = 'LBDIG5LC'
# TARGET_LB_COLUMN = 'LBDIG2LC'
MCQ_COLUMNS = ["MCQ160C", "MCQ010", "MCQ050", "MCQ092", "MCQ140"]
MCQ_MISSING_VALUE_REPLACEMENT = 4 # Value to replace NaNs in MCQ columns
RANDOM_SEED = 42
N_ITER_RANDOM_SEARCH = 200 # INCREASED from 100
CV_FOLDS = 5 # Number of cross-validation folds

def load_data(file_path):
    """Loads data from a CSV file."""
    print(f"--- 1. Loading Data from '{file_path}' ---")
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
        print(f"Original data shape: {data.shape}")
        # print("\n--- Original Data Sample (First 5 Rows) ---")
        # print(data.head())
        # print("\n--- Data Info ---")
        # data.info()
        return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please check the path.")
        exit()
    except Exception as e:
        print(f"Error loading CSV: {e}")
        exit()

def preprocess_data(data, target_col, feature_cols, missing_val_replacement):
    """Preprocesses the data: handles missing values, selects features/target."""
    print(f"\n--- 2. Preprocessing Data ---")
    # Check if all MCQ columns are present
    missing_mcq_cols = [col for col in feature_cols if col not in data.columns]
    if missing_mcq_cols:
        print(f"Error: MCQ columns not found: {missing_mcq_cols}. Available: {list(data.columns)}")
        exit()
    if target_col not in data.columns:
        print(f"Error: Target column '{target_col}' not found. Available: {list(data.columns)}")
        exit()

    X_raw = data[feature_cols].copy()
    y_raw = data[target_col].copy()

    # a. Handle Target Variable (y_raw)
    print(f"Preprocessing target variable '{target_col}'...")
    print(f"Original missing values in '{target_col}': {y_raw.isnull().sum()}")
    y_raw[y_raw == 2] = np.nan # Mark value 2 as missing
    print(f"Missing values after marking 2 as NaN: {y_raw.isnull().sum()}")
    
    y = y_raw.dropna()
    X = X_raw.loc[y.index]
    print(f"Number of rows after dropping NaNs in target: {len(y)}")

    if len(y) == 0:
        print(f"Error: No valid target values for '{target_col}' after dropping NaNs.")
        exit()
    try:
        y = y.astype(int)
    except ValueError as e:
        print(f"Error converting target '{target_col}' to int. Details: {e}")
        exit()
    print(f"Unique values in processed target '{target_col}': {np.sort(y.unique())}")

    # b. Handle Feature Variables (X - MCQ columns)
    print(f"Preprocessing feature variables (MCQ columns)...")
    print(f"Strategy: Missing MCQ values (7, 9, NaN) will be replaced with '{missing_val_replacement}'.")
    
    for col in feature_cols:
        X[col] = X[col].replace([7, 9], np.nan) # Replace 7 and 9 with NaN first
        X[col] = pd.to_numeric(X[col], errors='coerce') # Ensure numeric for fillna
        X[col].fillna(missing_val_replacement, inplace=True)
        try:
            X[col] = X[col].astype(int)
        except ValueError as e:
            print(f"Error converting feature '{col}' to int. Values: {X[col].unique()}. Details: {e}")
            exit()
            
    print("Missing values in MCQ columns AFTER replacement (should be 0):")
    print(X.isnull().sum())
    # print("\nSample of processed features (X) (First 5 rows):")
    # print(X.head())
    return X, y

def train_and_tune_model(X, y, random_seed, n_iter_search, cv_folds):
    """Trains and tunes an XGBoost model using SMOTE and RandomizedSearchCV."""
    print(f"\n--- 3. Model Training and Hyperparameter Tuning ---")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_seed, stratify=y)
    print(f"Training set size: {X_train.shape[0]} samples, Test set size: {X_test.shape[0]} samples")
    if X_train.empty or y_train.empty:
        print("\nError: Training data is empty after splitting. Cannot train model.")
        exit()

    # Define the pipeline with SMOTE and XGBoost
    smote = SMOTE(random_state=random_seed)
    xgb_clf = xgb.XGBClassifier(random_state=random_seed, 
                                objective='binary:logistic', 
                                eval_metric='logloss'
                                )

    pipeline = ImbPipeline([
        ('smote', smote),
        ('classifier', xgb_clf)
    ])

    param_dist = {
        'smote__k_neighbors': [3, 5, 7, 9], # ADDED SMOTE tuning
        'classifier__n_estimators': [100, 200, 300, 400, 500, 600, 800, 1000, 1200], # EXPANDED
        'classifier__max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'classifier__learning_rate': [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25], # EXPANDED
        'classifier__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'classifier__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'classifier__gamma': [0, 0.1, 0.2, 0.3, 0.5, 1, 1.5, 2], # EXPANDED
        'classifier__reg_alpha': [0, 0.001, 0.005, 0.01, 0.05, 0.1],
        'classifier__reg_lambda': [0.1, 0.5, 1, 1.5, 2, 5, 10]
    }

    # Stratified K-Fold for cross-validation
    strat_k_fold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)

    print(f"Running RandomizedSearchCV with n_iter={n_iter_search}, cv={cv_folds} folds...")
    start_time = time.time()
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter_search,
        scoring='accuracy', # Could also use 'f1_weighted', 'roc_auc', etc.
        cv=strat_k_fold,
        n_jobs=-1, # Use all available processors
        verbose=1, # Set to 2 or higher for more messages
        random_state=random_seed
    )

    random_search.fit(X_train, y_train)
    end_time = time.time()
    print(f"RandomizedSearchCV fitting completed in {(end_time - start_time):.2f} seconds.")

    best_model = random_search.best_estimator_
    print("\nBest hyperparameters found:")
    # Print only classifier parameters
    for param, value in random_search.best_params_.items():
        if param.startswith('classifier__'):
            print(f"{param.replace('classifier__', '')}: {value}")
    print(f"Best cross-validation accuracy: {random_search.best_score_:.4f}")
    
    return best_model, X_test, y_test

def evaluate_model(model, X_test, y_test, feature_cols):
    """Evaluates the model on the test set."""
    print(f"\n--- 4. Model Evaluation on Test Set ---")
    if X_test.empty or y_test.empty:
        print("Warning: Test set is empty. Skipping evaluation.")
        return

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on Test Set: {accuracy:.4f}")

    print("\nClassification Report:")
    present_labels = sorted(list(set(y_test.unique()) | set(y_pred)))
    if not present_labels:
        print("No labels found in y_test or y_pred for classification report.")
    else:
        target_names = [f"Class {l}" for l in present_labels]
        print(classification_report(y_test, y_pred, labels=present_labels, target_names=target_names, zero_division=0))

    print("\nConfusion Matrix:")
    if not present_labels:
        print("No labels found for confusion matrix.")
    else:
        cm = confusion_matrix(y_test, y_pred, labels=present_labels)
        cm_df = pd.DataFrame(cm, index=[f"Actual {l}" for l in present_labels], columns=[f"Predicted {l}" for l in present_labels])
        print(cm_df)

    print("\nFeature Importances:")
    try:
        # Access the classifier step from the pipeline
        if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
            classifier_in_pipeline = model.named_steps['classifier']
            if hasattr(classifier_in_pipeline, 'feature_importances_'):
                importances = classifier_in_pipeline.feature_importances_
                feature_importances_df = pd.DataFrame(
                    {'Feature': feature_cols, 'Importance': importances}
                ).sort_values('Importance', ascending=False)
                print(feature_importances_df)
            else:
                print("Feature importances not available for the classifier in pipeline.")
        else:
            print("Could not access classifier in pipeline for feature importances.")
    except Exception as e:
        print(f"Error retrieving feature importances: {e}")

def main():
    """Main function to run the training and evaluation pipeline."""
    overall_start_time = time.time()
    
    data = load_data(CSV_FILE_PATH)
    X, y = preprocess_data(data, TARGET_LB_COLUMN, MCQ_COLUMNS, MCQ_MISSING_VALUE_REPLACEMENT)
    
    if X.empty or y.empty:
        print("Exiting due to empty features or target after preprocessing.")
        return

    best_model, X_test, y_test = train_and_tune_model(X, y, RANDOM_SEED, N_ITER_RANDOM_SEARCH, CV_FOLDS)
    evaluate_model(best_model, X_test, y_test, MCQ_COLUMNS)
    
    overall_end_time = time.time()
    print(f"\n--- Total script execution time: {(overall_end_time - overall_start_time):.2f} seconds ---")

if __name__ == '__main__':
    main()
