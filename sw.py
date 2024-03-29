import pandas
from joblib import load, dump
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, classification_report, mean_absolute_error, mean_squared_log_error
from fasttext_fe import create_ft_word_vectors
from os.path import exists
from os import remove


MODEL_FILE_PREFIX_CLF = "./models/sw_clf_"
MODEL_FILE_PREFIX_RGR = "./models/sw_rgr_"
REGRESSION_LABELS = ["funny", "useful", "cool"]
CLASSIFICATION_LABELS = ["stars"]
TARGET_LABELS = ["funny", "useful", "cool", "stars"]


def sw_train(filepath: str) -> None:

    GRID_PARAMS = {
    "max_depth": [2, 7, 8, 9, 10],
    "random_state": [42]
    }

    print("--- Training Non-parametric Model ----")
    
    print("1. Loading training data...")
    training = pandas.read_csv(filepath)
    training.dropna(inplace=True)
    training_subset = training.sample(n=50000,random_state=42)

    print("2. Classification - Performing grid search using cross validation on a training subset to find the best settings for each target label...")
    best_params = {}
    for target in CLASSIFICATION_LABELS:
        print(f"Classification Target: {target}")
        gs = GridSearchCV(estimator=DecisionTreeClassifier(), scoring="f1_micro", refit=True, param_grid=GRID_PARAMS, n_jobs=4)
        gs.fit(X=training_subset.drop(TARGET_LABELS, axis=1), y=training_subset[target])
        best_params[target] = gs.best_params_
        print(f"Best f1_micro: {gs.best_score_}")
        print(f"Best Parameters: {gs.best_params_}")
    
    print("3. Regression - Performing grid search using cross validation on a training subset to find the best settings for each target label...")
    for target in REGRESSION_LABELS:
        print(f"Regression Target: {target}")
        gs = GridSearchCV(estimator=DecisionTreeRegressor(), scoring="neg_mean_squared_error", refit=True, param_grid=GRID_PARAMS, n_jobs=4)
        gs.fit(X=training_subset.drop(TARGET_LABELS, axis=1), y=training_subset[target])
        best_params[target] = gs.best_params_
        print(f"Best neg_mean_squared_error: {gs.best_score_}")
        print(f"Best Parameters: {gs.best_params_}")

    print("4. Training the final classification models using the best parameters...")
    for target in CLASSIFICATION_LABELS:
        print(f"Target: {target}")
        model = DecisionTreeClassifier(**best_params[target])
        model = model.fit(X=training.drop(TARGET_LABELS, axis=1), y=training[target])
        print(f"Saving model as {MODEL_FILE_PREFIX_CLF}{target}.joblib")
        dump(model, f"{MODEL_FILE_PREFIX_CLF}{target}.joblib")
        print("")

    print("5. Training the final regression models using the best parameters...")
    for target in REGRESSION_LABELS:
        print(f"Target: {target}")
        model = DecisionTreeRegressor(**best_params[target])
        model = model.fit(X=training.drop(TARGET_LABELS, axis=1), y=training[target])
        print(f"Saving model as {MODEL_FILE_PREFIX_RGR}{target}.joblib")
        dump(model, f"{MODEL_FILE_PREFIX_RGR}{target}.joblib")
        print("")


def sw_predict(filepath: str):
    
    FE_OUTPUT = "data/test_fe.csv"
    WORD_VECTOR_SIZE = 25

    print(f"Loading test set...\n")
    test = pandas.read_csv(filepath, chunksize=100000)

    print(f"Creating FastText word vectors...")
    if exists(FE_OUTPUT):
        remove(FE_OUTPUT)
    create_ft_word_vectors(test, FE_OUTPUT, WORD_VECTOR_SIZE)

    print(f"Loading FastText vectors...")
    test = pandas.read_csv(FE_OUTPUT)
    test.dropna(inplace=True)

    print(f"Performing Predictions...")
    print(f"Classification predictions...")
    for label in CLASSIFICATION_LABELS:
        model: DecisionTreeClassifier = load(f"{MODEL_FILE_PREFIX_CLF}{label}.joblib")
        y_pred = model.predict(test.drop(TARGET_LABELS, axis=1))
        print(f"Label: {label}")
        print(f"Classification Report:\n{classification_report(y_pred=y_pred, y_true=test[label])}")
    print("")

    print(f"Regression predictions...")
    for label in REGRESSION_LABELS:
        model: DecisionTreeRegressor = load(f"{MODEL_FILE_PREFIX_RGR}{label}.joblib")
        y_pred = model.predict(test.drop(TARGET_LABELS, axis=1))
        print(f"Label: {label}")
        print(f"Mean Squared Error - MSE: {mean_squared_error(y_pred=y_pred, y_true=test[label])}")
        print(f"Mean Absolute Error - MAE: {mean_absolute_error(y_pred=y_pred, y_true=test[label])}")
        print(f"Mean Squared Log Error - MSLE: {mean_squared_log_error(y_pred=y_pred, y_true=test[label])}")
