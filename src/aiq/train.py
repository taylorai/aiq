import os
import sys
import fire
import time
import json
import itertools
import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier
import hashlib
import joblib
from rich.console import Console
from rich.status import Status
from rich.style import Style
import select
from .common import get_shared_status

# deterministic way to split into train and test that won't vary between runs / batch sizes
def is_test(line, test_size):
        return int(hashlib.md5(line.encode()).hexdigest(), 16) % int(1 / test_size) == 0

def read_stdin_timeout(timeout):
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if rlist:
        return sys.stdin.readline()
    return None

def train(
    model_path: str,
    n_classes: int,
    epochs: int = 1,
    label_field: str = "label",
    input_field: str = "embedding",
    batch_size: int = 32,
    test_size: float = 0.1,
    timeout: float = 10.0
):
    corrected_batch_size = int(batch_size * (1 / (1 - test_size)))
    model = PassiveAggressiveClassifier(average=True)
    class_dict: dict[str, int] = {}
    X: np.ndarray | None = None
    X_test: np.ndarray | None = None
    y_test: np.ndarray | None = None
    X_train: np.ndarray | None = None
    y_train: np.ndarray | None = None
    console = Console(file=sys.stderr)
    stdout = Console(file=sys.stdout)

    # iterate over batches of lines from stdin
    status = Status("Starting training...", console=console, spinner_style=Style(color="magenta"))
    with status:
        batch = []
        last_input_time = time.time()
        i = 0
        while True:
            line = read_stdin_timeout(0.1)  # Small timeout for responsiveness

            if line:
                batch.append(line)
                last_input_time = time.time()

            if len(batch) >= corrected_batch_size or (not line and time.time() - last_input_time > timeout):
                if not batch:
                    break  # No more input and timeout reached

                # Process the batch
                data = [json.loads(line) for line in batch]
                is_test_mask = [is_test(line, test_size) for line in batch]

                X = np.array([d[input_field] for d in data])
                y_str: list[str] = [d[label_field] for d in data]

                for unique_label in set(y_str):
                    if unique_label not in class_dict:
                        class_dict[unique_label] = len(class_dict)
                        if len(class_dict) > n_classes:
                            raise RuntimeError("More classes found in data than specified in n_classes.")

                y_int = [class_dict[label] for label in y_str]
                y = np.array(y_int)

                X_test_subset, y_test_subset = X[is_test_mask], y[is_test_mask]
                X_train_subset, y_train_subset = X[~np.array(is_test_mask)], y[~np.array(is_test_mask)]

                if X_test is None:
                    X_test, y_test = X_test_subset, y_test_subset
                else:
                    assert y_test is not None, "y_test is None"
                    X_test = np.concatenate([X_test, X_test_subset], axis=0)
                    y_test = np.concatenate([y_test, y_test_subset], axis=0)

                model.partial_fit(X_train_subset, y_train_subset, classes=list(range(n_classes)))
                i += 1
                # print validation accuracy
                get_shared_status()
                input_size = os.environ.get("AIQ_INPUT_SIZE", None)
                if input_size is not None:
                    total_steps = int(input_size) // corrected_batch_size
                else:
                    total_steps = "???"
                accuracy = model.score(X_test, y_test)
                status.update(f"[bold]Training Step:[/bold] {i:>3}/{total_steps} | [bold]Test Accuracy = [green]{accuracy:.3f}[/green][/bold]")

                batch = []

            if not line and time.time() - last_input_time > timeout:
                break  # End of epoch or input

    # evaluate the model
    assert X_test is not None, "No test data provided"
    print(f"Evaluating on {len(X_test)} held-out examples...")
    preds = model.predict(X_test)
    # print([(pred, gold) for pred, gold in zip(preds, y_test)])
    assert y_test is not None, "y_test is None"
    accuracy = sum(pred == gold for pred, gold in zip(preds, y_test)) / len(y_test)
    print("Validation accuracy: {:.3f}".format(accuracy))
    print(f"Saving model to {model_path}...")

    # save the model
    assert X is not None, "X is None"
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
