import numpy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, \
    Trainer
from sklearn import metrics
from datasets import load_dataset
from transformers import pipeline
import pandas as pd
from os import path
from sklearn.metrics import mean_squared_error, classification_report, mean_absolute_error, mean_squared_log_error

REGRESSION_LABELS = ["funny", "useful", "cool"]
TARGET_LABELS = ["funny", "useful", "cool", "label"]
EXCESS_LABELS = ["__index_level_0__", "Unnamed: 0"]


def bh_predict(filepath: str = "data/test.csv"):
    test = pd.read_csv(filepath)
    test.dropna(inplace=True)
    test["stars"] = test["stars"].apply(lambda x: int(x) - 1)
    for label in TARGET_LABELS:
        _predict(test, label)


def _predict(test: pd.DataFrame, label: str):
    m_path = f"models/bh_regress_{label}"
    if label == "stars":
        m_path = "models/bh_classify"
    model = pipeline(task='text-classification', model=m_path)
    print(f"Loading test set...\n")

    x = test["text"].tolist()
    y_pred = model(x)
    print(f"Label: {label}")
    if label == "stars":
        print(f"Classification Report:\n{classification_report(y_pred=y_pred, y_true=test[label])}")
    else:
        print(f"Mean Squared Error - MSE: {mean_squared_error(y_pred=y_pred, y_true=test[label])}")
        print(f"Mean Absolute Error - MAE: {mean_absolute_error(y_pred=y_pred, y_true=test[label])}")
        print(f"Mean Squared Log Error - MSLE: {mean_squared_log_error(y_pred=y_pred, y_true=test[label])}")


def met(pred):
    predict, labels = pred
    predict = numpy.argmax(predict, axis=1)
    precision = metrics.precision_score(labels, predict, average='weighted')
    recall = metrics.recall_score(labels, predict, average='weighted')
    f1 = metrics.f1_score(labels, predict, average='weighted')
    ac = metrics.accuracy_score(labels, predict)
    return {
        "Accuracy": ac,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


def met_regress(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)

    mse = metrics.mean_squared_error(labels, logits)
    rmse = metrics.mean_squared_error(labels, logits, squared=False)
    mae = metrics.mean_absolute_error(labels, logits)
    r2 = metrics.r2_score(labels, logits)
    smape = 1 / len(labels) * numpy.sum(2 * numpy.abs(logits - labels) / (numpy.abs(labels) + numpy.abs(logits)) * 100)

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "smape": smape}


def classify_train(train_df):
    train_df = train_df.remove_columns(["useful", "cool", "funny"])
    train_df = train_df.rename_column("stars", "label")
    train_df = train_df.train_test_split(test_size=0.2)
    model_path = "distilbert-base-uncased"
    if path.exists("./models/bh_classify"):
        model_path = "./models/bh_classify"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def pre(text):
        return tokenizer(text['text'], truncation=True)

    token_train = train_df["train"].map(pre, batched=True)
    token_val = train_df["test"].map(pre, batched=True)
    dc = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=5, )

    train_arg = TrainingArguments(
        output_dir="star_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model='rmse'
    )

    trainer = Trainer(
        model=model,
        args=train_arg,
        train_dataset=token_train,
        eval_dataset=token_val,
        tokenizer=tokenizer,
        data_collator=dc,
        compute_metrics=met_regress,
    )

    trainer.train()
    print(trainer.evaluate())
    trainer.save_model("models/bh_classify")


def regression_train(train_df, label):
    for target in TARGET_LABELS:
        if target != label:
            train_df = train_df.remove_columns([target])
    train_df = train_df.rename_column(label, "label")
    train_df = train_df.train_test_split(test_size=0.2)
    model_path = "distilbert-base-uncased"
    if path.exists(f"models/bh_regress_{label}"):
        model_path = f"models/bh_regress_{label}"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def pre(text):
        return tokenizer(text['text'], truncation=True)

    token_train = train_df["train"].map(pre, batched=True)
    token_val = train_df["test"].map(pre, batched=True)
    dc = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=1)

    train_arg = TrainingArguments(
        output_dir=f"{label}_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=train_arg,
        train_dataset=token_train,
        eval_dataset=token_val,
        tokenizer=tokenizer,
        data_collator=dc,
        compute_metrics=met,
    )

    trainer.train()
    print(trainer.evaluate())
    trainer.save_model(f"models/bh_regress_{label}")


def bh_train(filepath="data/small_neural_train.csv"):
    train_df = load_dataset("csv", data_files=filepath)
    train_df = train_df["train"]
    names = train_df.column_names
    for label in EXCESS_LABELS:
        if label in names:
            train_df = train_df.remove_columns([label])
    print("Training stars")
    classify_train(train_df)
    for target in REGRESSION_LABELS:
        print(f"Training {target}")
        regression_train(train_df, target)


bh_predict()
