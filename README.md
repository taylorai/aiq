# aiq

![gif of aiq in the terminal](https://cdn.trytaylor.ai/aiq.gif)

`aiq` is a no-frills package for embeddings and text classification on the command line, inspired by the power of `jq`. It does 4 things:
- `aiq label`: Use LLM APIs to label a stream of texts
- `aiq embed`: Compute embeddings on a stream of texts
- `aiq train`: Train a text classifier (linear model) on a stream of embedded texts with labels
- `aiq classify`: Classify a stream of unlabeled text embeddings

These commands can operate on text and JSONL files, but they can also **read from stdin.** This means that you can them together: for example, you can use a single command to stream a text file in to be labeled, pipe the labeled data through an embedding model, and finally pipe the embedded, labeled training data through classifier training. (See the Quickstart below to learn how!)

## Quickstart

To use, install with pip. This will install dependencies, and the aiq command-line interface.
```bash
pip install aiq-cli
```

To use `aiq label`, you'll also need an OpenAI key. The other commands can be used on their own with no API key, as they run locally on your computer. Set the key as an environment variable:

```bash
export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

For this quickstart, we include an example dataset and labels file (`recipes.txt` and `label_options.yaml`) to try in the `examples` folder. After downloading these files, you can run the following command to train a model to classify recipe names into breakfast, lunch/dinner, dessert, etc.

```bash
aiq label --file recipes.txt --label-options-file label_options.yaml | aiq embed | aiq train --model_path model.joblib --n-classes 10
```

This uses an LLM to label the text in `unlabeled.txt`, with the label options from `label_options.yaml`. The text fields are embedded using an embedding model (runs locally on CPU). Finally, a passive-aggressive classifier is trained using the labels and embeddings. The resulting model will be saved to `model.joblib`.

You can then use `aiq classify` to run the model.

```bash
echo '{"text": "Maple-bacon and blueberry muffins"}' | aiq embed | aiq classify --model_path model.joblib
```

...which outputs the classified text:

```json
{"text": "Maple-bacon and blueberry muffins", "label": "breakfast"}
```

You'll also get a warning about loading the model, which is a reminder that it's not safe to load `aiq` models from untrusted sources. You can disable this warning by setting the `--no-warn` flag for `aiq classify`.

## Examples

### Label unlabeled data
The `aiq label` command can read text or JSON from a file or from `stdin`. If you just want to label data and save the result, you can pipe the output directly to a file.

```bash
aiq label --file recipes.txt --label-options-file label_options.yaml > labeled.jsonl
```

### Embed texts
Embed the `text` field in `data.jsonl` and write the embeddings to `embeddings.jsonl`:
```bash
cat data.jsonl | aiq embed --input_field text --progress > embeddings.jsonl
```

The `--progress` flag will show a running count of how many texts have been embedded. It should not be enabled if you're piping the output to another process, as the status indicators can interfere with each other.

### Use labeled data to train a classifier
Train a classifier with input in the `text` field and labels in the `label` field of `train_data.jsonl` and save it to `classifier.model`.

```bash
cat train_data.jsonl |
aiq embed --input_field text |
aiq train --label_field label --n_classes 2 --model_path classifier.model
```

Note that "text" and "label" are the default fields for the text and label, so these flags can be omitted.

### Inference with a classifier
Use the classifier to classify unlabeled data in `unlabeled_data.txt` and write the predictions to `predictions.jsonl`. By default, `aiq embed` expects JSON, so we have to set the `input-type` to be "text".

```bash
cat unlabeled_data.txt | aiq embed --input-type text |
aiq classify --model-path classifier.model --label-field prediction > predictions.jsonl
```

### Save intermediate outputs
Since `aiq` commands use `stdin`/`stdout` to communicate with each other, it's easy to combine them with other command-line utilities. For instance, if you don't just want the final output, you can use `tee` to sink intermediate outputs to a file:

```bash
cat unlabeled_data.txt | aiq embed --input-type text | tee embeddings.jsonl |
aiq classify --model-path classifier.model --label-field prediction > predictions.jsonl
```


### Chaining with `curl` and `jq`
You can also combine `aiq` with `curl` and `jq` since it natively reads JSON from `stdin`. This allows you to, for example, fetch some remote JSON, and then compute embeddings on it. Here, we're grabbing 10 randomly-generated beers and computing the text embedding on their names:
```bash
curl 'https://random-data-api.com/api/v2/beers?size=10' |
jq -c '.[]' |
aiq embed --input_field name > embeddings.jsonl
```

### Train a classifier on a remote dataset
Or, train a classifier to identify topics on some LLM fine-tuning data. We'll use `jq` to concatenate the "input" and "output" fields into one "text" field, then pass it to `aiq` to embed and train:
```bash
curl https://gist.githubusercontent.com/andersonbcdefg/a4c8483bd7ffd349e685a6c04660c179/raw/ff7c18b8530982312dafe5db750177fb3e8be186/topics.jsonl |
jq  -c '{text: (.input + " " + .output), topic: .topic}' |
aiq embed |
aiq train --label_field topic --n_classes 8 --model_path topics.model
```

### Command-line inference with the trained model
You can then infer topics on some new data:
```bash
echo '{"input": "What is the capital of Italy?", "output": "The capital of Italy is Rome."}' |
jq -c '{text: (.input + " " + .output)}' |
aiq embed |
aiq classify --model_path topics.model
```

And we get the output:
```json
{
    "text": "What is the capital of Italy? The capital of Italy is Rome.",
    "label": "world_knowledge"
}
```

## API Reference
Here's an exhaustive list of arguments for each `aiq` command. Note that for these CLI arguments, hyphens and underscores are interchangeable.

### `aiq label`
Label text inputs with an LLM. Supports raw text, or JSON objects.

Flags:
- `--input_type`: Whether the input stream is raw texts ("text"), or JSON ("json"). If raw texts, then `--input_field` will be ignored.
- `--input_field`: If reading JSON, specifies the field to use as the input for labeling. The default value is "text".
- `--label_options`: You can use this to directly provide inline label options. For instance: `--label_options '{"pos": "positive sentiment", "neg": "negative sentiment"}'`. This is null by default; it's recommended to use `--label_options_file` to keep the command simpler.
- `--label_options_file`: CSV, JSON, JSONL, or YAML with labels and (optionally) their definitions/descriptions. The column with the labels should be called `label`, and the column with the definitions/descriptions should be called `description`. For YAML/JSON, you may also provide a file where the labels are the keys, and the descriptions are the values.
- `--output_field`, `-o`: The field to put the LLM label in. Defaults to "label".
- `--model`: Name of the LLM to use for labeling. This is passed directly to the OpenAI client, so it supports any OpenAI chat model. You can also use any OpenAI-compatible API by setting the `api_base_url` (see below).
- `--file`, `-f`: If provided, reads from a file (can be text or JSON). Otherwise, reads from `stdin`, so it has to be chained.
- `--max_concurrency`: How many LLM completions can be running at once. You can increase this depending on your rate limits; it's 10 by default.
- `--api_key`: You can use this flag to pass your OpenAI (or OpenAI-compatible provider) API key, but it's recommended to use an environment variable instead.
- `--api_base_url`: You can use this flag (or the OPENAI_BASE_URL environment variable) to pass a different API URL from the default "https://api.openai.com/v1". This allows you to use other OpenAI-compatible LLM providers like TogetherAI, self-hosted vLLM or Ollama, etc.
- `--skip_errors`: Default False. If true, data points where an error happens will just be skipped and won't be put in the output stream. The benefit is that your process won't stop if an error occurs; the downside is your output may be missing some data from the input.
- `--progress`, `p`: Whether to show progress indicator for labeling. Do not enable this if you're piping the output to another `aiq` command; the status indicators interfere.

### `aiq embed`
Compute embeddings for a text field in a stream of texts or JSON objects.

Flags:
- `--input_type`: Whether the input is a stream of texts ("text"), or JSON objects ("json"). By default, it's "json", which makes it compatible with the output of `aiq label`.
- `--input_field`: If `input_type` is "json", the field to embed. Ignored for text input.
- `--output_field`, `-o`: The field in the JSON objects to write the embeddings to. Defaults to `embedding`.
- `--model_name`, `-m`: The name of the embedding model to use. Uses `snowflake-xs` by default, supports all models in the [`onnx_embedding_models`](https://github.com/taylorai/onnx_embedding_models) package.
- `--skip_errors`, `-s`: If true, skip over any errors that occur while embedding--inputs that cause errors will just not appear in the output. Otherwise, raise an exception. Defaults to `false`.
- `--progress`, `-p`: If true, will show progress in the console. Defaults to `false`, as the progress from embeddings can interfere with other progress bars and `embed` is designed to be chained.
- `--file`, `-f`: If provided, will read from the file instead of stdin. Supports `.jsonl` and `.txt`.


### `aiq train`
Train a text classifier on a stream of JSON objects with embeddings. Uses the PassiveAggressiveClassifier from `scikit-learn` for incremental learning, which means that the entire dataset never needs to be materialized in memory.

Flags:
- `--model_path`, `-m`: The path to save the model to.
- `--n_classes`, `-n`: The number of classes to train on. The trainer will automatically identify the label names during training. However, will throw an error if more unique labels than `n_classes` are encountered.
- `--label_field`, `-l`: The field in the JSON objects to use as the label. This should be a string field, and it defaults to "label", the default output of `aiq label`.
- `--input_field`, `-i`: The field in the JSON objects (which should be the embedding, a list of floats) to use as the input. Defaults to "embedding", the default output field of `aiq embed`.
- `--batch_size`, `-b`: The batch size to use for training. Defaults to 32.
- `--test_size`, `-t`: The proportion of the data to use for estimating out-of-sample accuracy. Defaults to 0.1.

### `aiq classify`
Classify a stream of JSON objects using their embeddings. Uses the model from `aiq train`.

Flags:
- `--model_path`, `-m`: The path to load the model from.
- `--label_field`, `-l`: The field name for the predicted label in the output JSON. Can be different than the label field used for training.
- `--input_field`, `-i`: The field in the JSON objects (which should be the embedding, a list of floats) to use as the input. Defaults to "embedding", which is the default output field of `embed`.
- `--remove_input`, `-r`: If this flag is set, remove the input (i.e. embedding) field from the output JSON. Embeddings can get really large and take up a lot of space, and you might not need it in the final result. Enabled by default.
- `--skip_errors`, `-s`: If true, skip over any errors that occur while embedding--inputs that cause errors will just not appear in the output. Otherwise, raise an exception. Defaults to `false`.
- `--no-warn`: If set, do not show the error about loading models from untrusted sources.

## Use from Python
This tool is written in Python, and it works fine as a Python library. You can import the `label`, `embed`, `train`, and `classify` functions from `aiq` and use them directly.
