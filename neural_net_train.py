import numpy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
from sklearn import metrics
from datasets import load_dataset

def pre(text):
    return tokenizer(text['text'], truncation=True)

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

if __name__ == "__main__":
    train_df = load_dataset("csv", data_files="data/small_neural_train.csv")
    train_df = train_df["train"]
    train_df = train_df.remove_columns(["__index_level_0__","useful", "cool", "funny"])
    train_df = train_df.train_test_split(test_size=0.2)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    token_train = train_df["train"].map(pre, batched=True)
    token_val = train_df["test"].map(pre, batched=True)
    dc = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5,
                                                               )

    train_arg = TrainingArguments(
        output_dir="star_model",
        learning_rate=2e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
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
    trainer.save_model("models/bh_classify")
