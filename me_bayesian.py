from joblib import load, dump
import pandas
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import BayesianRidge
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, classification_report, mean_absolute_error, mean_squared_log_error
from sklearn.pipeline import Pipeline
import sys

MODEL_FILE_PREFIX_CLF = "./models/me_clf_"
MODEL_FILE_PREFIX_FS = "./models/me_fs"
MODEL_FILE_PREFIX_RGR = "./models/me_rgr_"
REGRESSION_LABELS = ["funny", "useful", "cool"]
CLASSIFICATION_LABELS = ["stars"]
TARGET_LABELS = ["funny", "useful", "cool", "stars"]

RANDOM_STATE = 42

def bayesian_train(filepath: str, feature_selection: bool = False):

    _fs = "_fs" if feature_selection else ""
    feature_selection_params = {
        'selector__k': [5, 8, 10, 12, 15, 20]
    }

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

    training_data_subset: pandas.DataFrame = training_subset.drop(TARGET_LABELS, axis=1)
    training_data: pandas.DataFrame = training.drop(TARGET_LABELS, axis=1)

    if feature_selection:
        print("")
        print("Optional Feature Selection taking place")
        selector = SelectKBest(f_regression)
        clf = GaussianNB()
        pipe = Pipeline([
            ('selector', selector),
            ('classifier', clf)
        ])
        gs = GridSearchCV(pipe, param_grid=feature_selection_params, cv=5)
        gs.fit(X=training_subset.drop(TARGET_LABELS, axis=1), y=training_subset["stars"])
        # Print the best alpha value and corresponding score
        print("Best k value: ", gs.best_params_['selector__k'])
        print("Best score: ", gs.best_score_)
        selector = SelectKBest(f_regression, k = gs.best_params_['selector__k'])
        X_new = selector.fit_transform(training_data_subset,  y=training_subset["stars"])

        # Display the number of features selected, which features were selected, and their p-values.
        selected_features = training_data_subset.columns[selector.get_support()]

        print(f"Number of Features: {len(selected_features)}")
        print(f"Selected Features: {selected_features}")
        print(f"Feature P-Values: {selector.pvalues_}")
        print("")
        # create a new dataframe with only the selected features
        training_data_subset = pandas.DataFrame(X_new, columns=selected_features)
        training_data = pandas.DataFrame(training, columns=selected_features)
        dump(selector, f"{MODEL_FILE_PREFIX_FS}.joblib")

    print("2. Classification - Performing grid search using cross validation on a training subset to find the best settings for each target label...")
    best_params = {}
    for target in CLASSIFICATION_LABELS:
        print(f"Classification Target: {target}")
        gs = GridSearchCV(estimator=GaussianNB(), scoring="f1_micro", refit=True, param_grid=classification_grid_params, n_jobs=5)
        gs.fit(X=training_data_subset, y=training_subset[target])
        best_params[target] = gs.best_params_
        print(f"Best f1_micro: {gs.best_score_}")
        print(f"Best Parameters: {gs.best_params_}")

    print("3. Regression - Performing grid search using cross validation on a training subset to find the best settings for each target label...")
    for target in REGRESSION_LABELS:
        print(f"Regression Target: {target}")
        gs = GridSearchCV(estimator=BayesianRidge(), scoring="neg_mean_squared_error", refit=True, param_grid=regressor_grid_params)
        gs.fit(X=training_data_subset, y=training_subset[target])
        best_params[target] = gs.best_params_
        print(f"Best neg_mean_squared_error: {gs.best_score_}")
        print(f"Best Parameters: {gs.best_params_}")

    print("4. Training the final classification models using the best parameters...")
    for target in CLASSIFICATION_LABELS:
        print(f"Classification Target: {target}")
        clf = GaussianNB(**best_params[target])
        clf = clf.fit(X=training_data, y=training[target])
        print(f"Saving model to {MODEL_FILE_PREFIX_CLF}{target}{_fs}.joblib")
        dump(clf, f"{MODEL_FILE_PREFIX_CLF}{target}{_fs}.joblib")

    print("5. Training the final regression models using the best parameters...")
    for target in REGRESSION_LABELS:
        print(f"Regression Target: {target}")
        rgr = BayesianRidge(**best_params[target])
        rgr = rgr.fit(X=training_data, y=training[target])
        print(f"Saving model to {MODEL_FILE_PREFIX_RGR}{target}{_fs}.joblib")
        dump(rgr, f"{MODEL_FILE_PREFIX_RGR}{target}{_fs}.joblib")

def bayesian_predict(filepath: str, feature_selection: bool = False):
    _fs = "_fs" if feature_selection else ""
    print("--- Predicting Probabalistic Naive Bayes Model ----")
    
    print("1. Loading training data...")
    test = pandas.read_csv(filepath)
    test.dropna(inplace=True)
    print(f"Loading test set...")
    test = pandas.read_csv(filepath)
    test.dropna(inplace=True)

    test_data: pandas.DataFrame = test.drop(TARGET_LABELS, axis=1)

    print("Optional Feature Selection taking place")
    if feature_selection:
        selector: SelectKBest = load(f"{MODEL_FILE_PREFIX_FS}.joblib")
        selected_features = test_data.columns[selector.get_support()]
        test_data = pandas.DataFrame(test_data, columns=selected_features)

    print(f"2. Classification predictions")
    for label in CLASSIFICATION_LABELS:
        clf: GaussianNB = load(f"{MODEL_FILE_PREFIX_CLF}{label}{_fs}.joblib")
        y_pred = clf.predict(test_data)
        print(f"Label: {label}")
        print(f"Classification Report:\n{classification_report(y_pred=y_pred, y_true=test[label])}")
    print("")

    print(f"3. Regression predictions")
    for label in REGRESSION_LABELS:
        rgr: BayesianRidge = load(f"{MODEL_FILE_PREFIX_RGR}{label}{_fs}.joblib")
        y_pred = rgr.predict(test_data)
        print(f"Label: {label}")
        print(f"Mean Squared Error - MSE: {mean_squared_error(y_pred=y_pred, y_true=test[label])}")
        print(f"Mean Absolute Error - MAE: {mean_absolute_error(y_pred=y_pred, y_true=test[label])}")
