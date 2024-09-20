import os
import fire
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = "1"
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = "1"

from aiq.common import reset_shared_status
from aiq.train import train
from aiq.embed import embed
from aiq.classify import classify
from aiq.label import label

def main():
    reset_shared_status()
    fire.Fire({
        "label": label,
        "train": train,
        "embed": embed,
        "classify": classify
    })
