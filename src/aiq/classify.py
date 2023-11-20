import sys
import select
import json
from rich.console import Console
from rich.status import Status
import time
import joblib

def classify(
    model_path: str,
    label_field: str,
    input_field: str = "__embedding__",
    remove_embedding: bool = True
):
    console = Console(file=sys.stderr)
    stdout = Console(file=sys.stdout)
    console.print("[yellow]⚠️  Warning: Loading models uses pickle, which can execute arbitrary Python code. " +
          "If you don't trust the source of this model file, press CTRL + C to exit.[/yellow]")
    time.sleep(5)
    model_obj = joblib.load(model_path)
    model = model_obj["model"]
    label2idx = model_obj["class_dict"]
    idx2label = {v: k for k, v in label2idx.items()}

    with Status("", console=console) as status:
        examples_read = 0
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 5)
            if not ready:
                # No new data within the timeout period
                if examples_read == 0:
                    # If we haven't read any examples, wait a bit longer
                    continue
                else:
                    # If we've processed some data, assume we're done
                    break
            
            line = sys.stdin.readline()
            if not line:
                # End of file reached
                break

            # Process the line
            try:
                loaded = json.loads(line)
            except:
                console.print(f"[red]Error: Could not parse line {examples_read} as JSON. Example: {line} Skipping...[/red]")
            X = loaded[input_field]
            y = model.predict([X])[0]
            label = idx2label[y]
            loaded[label_field] = label
            if remove_embedding:
                del loaded[input_field]
            stdout.print(json.dumps(loaded), markup=False)
            examples_read += 1
            status.update(f"Classified {examples_read} / ??? examples...")
    
    console.print("Done.")



