"""Download the three official OpenCity traffic archives and verify their structure."""
import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path

CITIES = ["TrafficCD", "TrafficHZ", "TrafficSH"]
BASE = "https://huggingface.co/datasets/hkuds/OpenCity-dataset/resolve/main"

p = argparse.ArgumentParser()
p.add_argument("--root", default="data/open_city/raw")
p.add_argument("--cache", default="data/downloads")
a = p.parse_args()
root, cache = Path(a.root).resolve(), Path(a.cache).resolve()
root.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
for city in CITIES:
    archive = cache / f"{city}.zip"
    if not archive.exists():
        print(f"Downloading {city}...")
        urllib.request.urlretrieve(f"{BASE}/{city}.zip?download=true", archive)
    print(city, hashlib.sha256(archive.read_bytes()).hexdigest())
    destination = root
    destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            final = (destination / member.filename).resolve()
            if destination != final and destination not in final.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        zf.extractall(destination)
    npz = list(destination.rglob(f"{city}.npz"))
    adjacency = list(destination.rglob(f"{city}_rn_adj.npy"))
    if len(npz) != 1 or len(adjacency) != 1:
        raise RuntimeError(f"Unexpected {city} archive structure")
print(f"Ready: {root}")
