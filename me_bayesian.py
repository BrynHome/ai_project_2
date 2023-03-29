from joblib import load, dump
import pandas
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import BayesianRidge
from sklearn.naive_bayes import GaussianNB

MODEL_FILE_PREFIX_CLF = "./models/me_clf_"
MODEL_FILE_PREFIX_RGR = "./models/me_rgr_"
REGRESSION_LABELS = ["funny", "useful", "cool"]
CLASSIFICATION_LABELS = ["stars"]
TARGET_LABELS = ["funny", "useful", "cool", "stars"]

RANDOM_STATE = 42

def bayesian_train(filepath: str):

    classification_grid_params = {
        'var_smoothing': [1e-14, 1e-13, 1e-11, 1e-10, 1e-9]
    }

    regressor_grid_params = {
        'tol': [1e-4, 1e-3, 1e-2],
        'alpha_1': [1e-8, 1e-6, 1e-4],
        'lambda_1': [1e-8, 1e-6, 1e-4],
        'alpha_2': [1e-8, 1e-6, 1e-4],
        'lambda_2': [1e-8, 1e-6, 1e-4],
    }

    print("--- Training Probabalistic Naive Bayes Model ----")
    
    print("1. Loading training data...")
    training = pandas.read_csv(filepath)
    training.dropna(inplace=True)
    training_subset = training.sample(n=25000,random_state=RANDOM_STATE)

    print("2. Classification - Performing grid search using cross validation on a training subset to find the best settings for each target label...")
    best_params = {}
    for target in CLASSIFICATION_LABELS:
        print(f"Classification Target: {target}")
        gs = GridSearchCV(estimator=GaussianNB(), scoring="f1_micro", refit=True, param_grid=classification_grid_params, n_jobs=5)
        gs.fit(X=training_subset.drop(TARGET_LABELS, axis=1), y=training_subset[target])
        best_params[target] = gs.best_params_
        print(f"Best f1_micro: {gs.best_score_}")
        print(f"Best Parameters: {gs.best_params_}")

    print("3. Regression - Performing grid search using cross validation on a training subset to find the best settings for each target label...")
    for target in REGRESSION_LABELS:
        print(f"Regression Target: {target}")
        gs = GridSearchCV(estimator=BayesianRidge(), scoring="neg_mean_squared_error", refit=True, param_grid=regressor_grid_params)
        gs.fit(X=training_subset.drop(TARGET_LABELS, axis=1), y=training_subset[target])
        best_params[target] = gs.best_params_
        print(f"Best neg_mean_squared_error: {gs.best_score_}")
        print(f"Best Parameters: {gs.best_params_}")

    print("4. Training the final classification models using the best parameters...")
    for target in CLASSIFICATION_LABELS:
        print(f"Classification Target: {target}")
        clf = GaussianNB(**best_params[target])
        clf = clf.fit(X=training.drop(TARGET_LABELS, axis=1), y=training[target])
        print(f"Saving model to {MODEL_FILE_PREFIX_CLF}{target}.joblib")
        dump(clf, f"{MODEL_FILE_PREFIX_CLF}{target}.joblib")

    print("5. Training the final regression models using the best parameters...")
    for target in REGRESSION_LABELS:
        print(f"Regression Target: {target}")
        rgr = BayesianRidge(**best_params[target])
        rgr = rgr.fit(X=training.drop(TARGET_LABELS, axis=1), y=training[target])
        print(f"Saving model to {MODEL_FILE_PREFIX_RGR}{target}.joblib")
        dump(rgr, f"{MODEL_FILE_PREFIX_RGR}{target}.joblib")
