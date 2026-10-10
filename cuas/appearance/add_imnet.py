import argparse
import hashlib
import io
import os

from PIL import Image

SRC = [
    ("mlnomad/imnet1k_warplane_military_plane", "drone", 1300),
    ("mlnomad/imnet1k_airliner", "drone", 500),
    ("mlnomad/imnet1k_American_egret_great_white_heron_Egretta_albus", "bird", 700),
    ("mlnomad/imnet1k_little_blue_heron_Egretta_caerulea", "bird", 500),
]


def split_of(key):
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
    return "train" if h < 75 else ("val" if h < 87 else "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/cuas_data"))
    ap.add_argument("--crops", default=os.path.expanduser("~/cuas_data/crops"))
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    for repo, cls, cap in SRC:
        tag = repo.split("imnet1k_")[1][:20]
        f = hf_hub_download(repo, "data/train-00000-of-00001.parquet", repo_type="dataset",
                            local_dir=f"{a.root}/imnet/{tag}")
        col = pq.read_table(f, columns=["image"]).column("image").to_pylist()[:cap]
        n = {}
        for i, rec in enumerate(col):
            sp = split_of(f"{tag}{i}")
            d = f"{a.crops}/{sp}/{cls}"
            os.makedirs(d, exist_ok=True)
            im = Image.open(io.BytesIO(rec["bytes"])).convert("RGB")
            im.thumbnail((256, 256))
            im.save(f"{d}/imnet_{tag}_{i}.jpg", quality=90)
            n[sp] = n.get(sp, 0) + 1
        print(repo, "->", cls, n, flush=True)


if __name__ == "__main__":
    main()
