import fire

from aiq.train import train
from aiq.embed import embed
from aiq.classify import classify

def main():
    fire.Fire({"train": train, "embed": embed, "classify": classify})