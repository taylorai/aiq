import sys
import fire
import json
import itertools
import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier
import hashlib
import joblib
from rich.console import Console
from rich.status import Status

# deterministic way to split into train and test that won't vary between runs / batch sizes
def is_test(line, test_size):
        return int(hashlib.md5(line.encode()).hexdigest(), 16) % int(1 / test_size) == 0

def train(
    label_field: str,
    model_path: str,
    n_classes: int,
    epochs: int = 1,
    input_field: str = "__embedding__",
    batch_size: int = 225,
    test_size: float = 0.1,
):
    corrected_batch_size = int(batch_size * (1 / (1 - test_size)))
    model = PassiveAggressiveClassifier(average=True)
    class_dict = {}
    X_test = None
    y_test = None
    X_train = None
    y_train = None

    # iterate over batches of lines from stdin
    print("\nTraining model...")
    status = Status("")
    with status:
        for i, lines in enumerate(iter(lambda: list(itertools.islice(sys.stdin, corrected_batch_size)), [])):
            # parse the lines into a list of dicts
            data = [json.loads(line) for line in lines]
            is_test_mask = [is_test(line, test_size) for line in lines]

            # extract the input and label fields
            X = np.array([d[input_field] for d in data]) # this is a list of floats
            y = [d[label_field] for d in data]
            
            for unique_label in set(y):
                if unique_label not in class_dict:
                    class_dict[unique_label] = len(class_dict)
                    if len(class_dict) > n_classes:
                        raise RuntimeError(
                            "More classes found in data than specified in n_classes."
                        )
            y = np.array([class_dict[label] for label in y])

            X_test_subset, y_test_subset = X[is_test_mask], y[is_test_mask]
            X_train_subset, y_train_subset = X[~np.array(is_test_mask)], y[~np.array(is_test_mask)]
            X_test = np.concatenate((X_test, X_test_subset), axis=0) if X_test is not None else X_test_subset
            y_test = np.concatenate((y_test, y_test_subset), axis=0) if y_test is not None else y_test_subset

            # fit the model
            model.partial_fit(X_train_subset, y_train_subset, classes=list(range(n_classes)))
            
            # print validation accuracy
            accuracy = model.score(X_test, y_test)
            status.update(f"Step {i + 1} / ??? | Accuracy={accuracy:.3f}")

    # evaluate the model
    print("Evaluating on {} held-out examples...".format(len(X_test)))
    preds = model.predict(X_test)
    # print([(pred, gold) for pred, gold in zip(preds, y_test)])
    accuracy = sum(pred == gold for pred, gold in zip(preds, y_test)) / len(y_test)
    print("Validation accuracy: {:.3f}".format(accuracy))
    print(f"Saving model to {model_path}...")
    
    # save the model
    model_obj = {
        "model": model,
        "class_dict": class_dict,
        "input_field": input_field,
        "label_field": label_field,
        "embedding_dim": len(X[0]),
        "accuracy": accuracy,
    }
    joblib.dump(model_obj, f"{model_path}")
    print("Done.")

if __name__ == "__main__":
    fire.Fire(train)