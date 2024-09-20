import os
import sys
import rich
import json
import yaml
import asyncio
import pandas as pd
from typing import Optional, Literal
from aiq.common import set_shared_status, count_lines
from openai import AsyncOpenAI
from rich.console import Console
from rich.status import Status
from rich.style import Style
import contextlib

async def get_label(
    text: str,
    label_options: dict[str, str],
    client: AsyncOpenAI,
    model: str
):
    prompt = (
            f"Classify the following text into one of the following categories:\n"
        f"{json.dumps(label_options, indent=2)}\n\n"
        f"Text:\n{text}\n\n"
        f"Respond with just the category name, no prelude or commentary required."
    )
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=8,
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

def read_options_file(file: str) -> dict[str, str]:
    if file.endswith(".csv"):
        df = pd.read_csv(file)
        return {row["label"]: row["description"] for row in df.to_dict(orient="records")}
    elif file.endswith("jsonl"):
        df = pd.read_json(file, lines=True, orient="records")
        return {row["label"]: row["description"] for row in df.to_dict(orient="records")}
    elif file.endswith(".json"):
        with open(file, "r") as f:
            content = json.load(f)
            if isinstance(content, list):
                return {row["label"]: row["description"] for row in content}
            elif isinstance(content, dict):
                return content
            else:
                raise ValueError("Invalid JSON file format")
    elif file.endswith(".yaml") or file.endswith(".yml"):
        with open(file, "r") as f:
            content = yaml.safe_load(f)
            if isinstance(content, list):
                return {row["label"]: row["description"] for row in content}
            elif isinstance(content, dict):
                return content
            else:
                raise ValueError("Invalid YAML file format")
    elif file.endswith(".txt"):
        labels = open(file, "r").readlines()
        return {label.strip(): "" for label in labels}
    else:
        raise ValueError(f"Unsupported file format: {file}")

def label(
    input_type: Literal["json", "text"] = "text",
    input_field: str | None = None,
    label_options: list[str] | dict[str, str] | None = None,
    label_options_file: Optional[str] = None, # must be csv, json, yaml
    output_field: str = "label",
    model: str = "gpt-4o-mini",
    file: Optional[str] = None,
    max_concurrency: int = 10,
    api_key: Optional[str] = None,
    api_base_url: Optional[str] = None,
    skip_errors: bool = False,
    progress: bool = False # Do not turn on if piping to another process; they'll interfere
):
    if input_type == "json" and input_field is None:
        raise ValueError("input_type is 'json' but input_field is not provided")
    if input_type == "text" and input_field is not None:
        sys.stderr.write("Warning: input_type is 'text' but input_field is provided. Ignoring input_field.\n")
    # set up label options
    if label_options is None:
        if label_options_file is None:
            raise ValueError("label_options or label_options_file must be provided")
        label_options = read_options_file(label_options_file)
    if isinstance(label_options, list):
        label_options_dict = {x: "" for x in label_options}
    elif isinstance(label_options, dict):
        label_options_dict = label_options
    else:
        raise ValueError("label_options must be a list or dict")
    console = Console(file=sys.stderr)
    stdout = Console(file=sys.stdout)
    examples_read = 0

    # Initialize the asynchronous OpenAI client
    client = AsyncOpenAI(
        api_key=api_key or os.environ["OPENAI_API_KEY"],
        base_url=api_base_url or os.environ.get("OPENAI_BASE_URL", None)
    )

    # Determine the input source
    if file is None:
        file_handle = sys.stdin
    else:
        # count lines in file
        total_lines = count_lines(file)
        set_shared_status("AIQ_INPUT_SIZE", str(total_lines))
        file_handle = open(file, "r")

    async def async_label():
        nonlocal examples_read
        semaphore = asyncio.Semaphore(max_concurrency)  # Limit concurrent tasks
        tasks = []

        # Define a coroutine to process each line
        async def process_line(line: str):
            nonlocal examples_read
            try:
                if input_type == "json":
                    input_json = json.loads(line)
                    input_text = input_json[input_field]
                else:
                    input_json = {"text": line}
                    input_text = line
                if not input_text:
                    return
                label_result = await get_label(
                    input_text, label_options_dict, client, model
                )
                output_json = {
                    **input_json,
                    output_field: label_result
                }
                stdout.print(json.dumps(output_json), markup=False, soft_wrap=True)
                examples_read += 1
                if progress and status is not None:
                    status.update(f"Labeled {examples_read} examples...")
            except Exception as e:
                if not skip_errors:
                    raise RuntimeError(f"Error processing line: {line}. Error: {str(e)}") from e

        # Create a status context if progress is enabled
        status_context = Status(
            "", console=console, spinner_style=Style(color="purple")
        ) if progress else contextlib.nullcontext()

        total_lines = 0
        with status_context as status:
            # Iterate over each line and schedule tasks
            for line in file_handle:
                if not line.strip():
                    continue  # Skip empty lines
                # Acquire semaphore before scheduling the task
                await semaphore.acquire()
                task = asyncio.create_task(process_line(line))
                task.add_done_callback(lambda t: semaphore.release())
                tasks.append(task)

            # Wait for all tasks to complete
            if tasks:
                if progress and status is not None:
                    status.update(f"Processing {len(tasks)} examples...")
                await asyncio.gather(*tasks)

        # update env variable
        set_shared_status("AIQ_INPUT_SIZE", str(total_lines))

    # Run the asynchronous helper function
    asyncio.run(async_label())

    # Close the file handle if it's not stdin
    if file is not None:
        file_handle.close()
