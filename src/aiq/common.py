import os
import time
import dotenv

def count_lines(filename):
    with open(filename, 'rb') as f:
        return sum(1 for _ in f)

def reset_shared_status():
    with open("/tmp/aiq.status", "w") as f:
        f.write("")

def get_shared_status():
    dotenv.load_dotenv("/tmp/aiq.status")

def set_shared_status(key, value):
    with open("/tmp/aiq.status", "a") as f:
        f.write(f"{key}={value}\n")
