# aiq
it's like jq, but with ✨AI✨

aiq is a no-frills package for embeddings and text classification on the command line. It does 3 things:
- Compute embeddings for a text field in a stream of JSON objects
- Train a text classifier on a stream of JSON objects with embeddings
- Classify a stream of JSON objects with embeddings

All of these operations work on JSONL files, but can also read from stdin. This means that you can chain together the embedding step and the training/inference step to do them in parallel.

## Command-Line Examples

### Embed a text field
Embed the `text` field in `data.jsonl` and write the embeddings to `embeddings.jsonl`:
```bash
cat data.jsonl | 
aiq embed --input_field text --progress > embeddings.jsonl
```

### Train a classifier
Train a classifier with input in the `text` field and labels in the `label` field of `train_data.jsonl` and save it to `classifier.model`:
```bash
cat train_data.jsonl | 
aiq embed --input_field text | 
aiq train --label_field label --n_classes 2 --model_path classifier.model
```

### Inference with a classifier
Use the classifier to classify unlabeled data in `unlabeled_data.jsonl` and write the predictions to `predictions.jsonl`:
```bash
cat unlabeled_data.jsonl | 
aiq embed --input_field text | 
aiq classify --model_path classifier.model --label_field prediction > predictions.jsonl
```

### Chaining with `curl` and `jq`
Because `aiq` reads from stdin, it also plays nice with other awesome command-line utilities, like `jq`. This allows you to, for example, fetch some remote JSON, and then compute embeddings on it. Here, we're grabbing 10 randomly-generated beers and computing the text embedding on their names:
```bash
curl 'https://random-data-api.com/api/v2/beers?size=10' | 
jq -c '.[]' | 
aiq embed --input_field name > embeddings.jsonl
```

### Training a classifier on a remote dataset
Or, train a classifier to identify topics on some LLM fine-tuning data. We'll use `jq` to concatenate the "input" and "output" fields into one "text" field, then pass it to `aiq` to embed and train:
```bash
curl https://gist.githubusercontent.com/andersonbcdefg/a4c8483bd7ffd349e685a6c04660c179/raw/ff7c18b8530982312dafe5db750177fb3e8be186/topics.jsonl | 
jq  -c '{text: (.input + " " + .output), topic: .topic}' | 
aiq embed --input_field text | 
aiq train --label_field topic --n_classes 8 --model_path topics.model
```

### Command-line inference with the trained model
You can then infer topics on some new data:
```bash
echo '{"input": "What is the capital of Italy?", "output": "The capital of Italy is Rome."}' | 
jq -c '{text: (.input + " " + .output)}' | 
aiq embed --input_field text | 
aiq classify --model_path topics.model --label_field prediction
```

And we get the output:
```json
{
    "text": "What is the capital of Italy? The capital of Italy is Rome.", 
    "prediction": "world_knowledge"
}
```

## Installation
To get started, install with pip:
```
pip install aiq @ git+https://github.com/andersonbcdefg/aiq.git
```

This should install the necessary packages and the aiq executable command-line utility.

## API Reference

### `aiq embed`
Compute embeddings for a text field in a stream of JSON objects.

Flags:
- `--input_field`, `-i`: The in the JSON objects to embed. This field should be a string.
- `--output_field`, `-o`: The field in the JSON objects to write the embeddings to. Defaults to `__embedding__`.
- `--model_name`, `-m`: The name of the embedding model to use. Defaults to `bge-micro-v2`. Other options are `bge-micro`, `gte-tiny`, and `gte-small`.
- `--skip_errors`, `-s`: If true, skip over any errors that occur while embedding--inputs that cause errors will just not appear in the output. Otherwise, raise an exception. Defaults to `false`.
- `--progress`, `-p`: If true, will show progress in the console. Defaults to `false`, as the progress from embeddings can interfere with other progress bars and `embed` is designed to be chained.
- `--file`, `-f`: If provided, will read from the file instead of stdin. Only supports JSONL.

### `aiq train`
Train a text classifier on a stream of JSON objects with embeddings. Uses the PassiveAggressiveClassifier from `scikit-learn` for incremental learning, which means that the entire dataset never needs to be materialized in memory.

Flags:
- `--label_field`, `-l`: The field in the JSON objects to use as the label. This field should be a string.
- `--model_path`, `-m`: The path to save the model to.
- `--n_classes`, `-n`: The number of classes to train on. The trainer will automatically identify the label names during training. However, will throw an error if more unique labels than `n_classes` are encountered.
- `--epochs`, `-e`: The number of epochs to train for. Currently this doesn't do anything, will add multi-epoch support later.
- `--input_field`, `-i`: The field in the JSON objects (which should be the embedding, a list of floats) to use as the input. Defaults to `__embedding__`, which is the default output field of `embed`.
- `--batch_size`, `-b`: The batch size to use for training. Defaults to 225.
- `--test_size`, `-t`: The proportion of the data to use for estimating out-of-sample accuracy. Defaults to 0.1.

### `aiq classify`
Classify a stream of JSON objects using their embeddings. Uses the model from `aiq train`.

Flags:
- `--model_path`, `-m`: The path to load the model from.
- `--label_field`, `-l`: The field name for the predicted label in the output JSON. Can be different than the label field used for training.
- `--input_field`, `-i`: The field in the JSON objects (which should be the embedding, a list of floats) to use as the input. Defaults to `__embedding__`, which is the default output field of `embed`.
- `--remove_embedding`, `-r`: If , remove the embedding field from the output JSON. Embeddings can get really large and take up a lot of space, and you might not need it in the final result. Defaults to `true`.

## Usage from Python
This library is written in Python, and it works fine as a library. You can import the `embed`, `train`, and `classify` functions from `aiq` and use them directly. They aren't doing anything super fancy, but this could still be a convenient set of utilities.