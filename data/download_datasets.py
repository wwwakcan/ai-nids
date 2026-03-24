"""
Dataset Downloader
==================
Usage:
  python data/download_datasets.py             # downloads NSL-KDD by default
  python data/download_datasets.py --dataset cicids
  python data/download_datasets.py --dataset all
"""

import argparse, os, urllib.request


NSL_KDD_FILES = {
    "KDDTrain+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
    "KDDTest+.txt":  "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
}


def download_nslkdd(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    print("Downloading NSL-KDD dataset...")
    for filename, url in NSL_KDD_FILES.items():
        dest = os.path.join(data_dir, filename)
        if os.path.exists(dest):
            print(f"  ✅ {filename} already exists — skipping")
            continue
        print(f"  Downloading {filename}...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, dest)
            size_mb = os.path.getsize(dest) / 1024 / 1024
            print(f"done ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"FAILED: {e}")
            print(f"    Manual download: http://www.unb.ca/cic/datasets/nsl.html")
    print()


def download_cicids(data_dir="data/cicids"):
    print("CICIDS2017 must be downloaded manually (requires registration).")
    print("  URL: https://www.unb.ca/cic/datasets/ids-2017.html")
    print(f"  Place CSV files in: {os.path.abspath(data_dir)}/")
    print()


def generate_sample_flows(data_dir="data", n=100):
    """Generate a small CSV demo file for testing the API."""
    import numpy as np
    import csv

    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "sample_flows.csv")
    if os.path.exists(path):
        print(f"  ✅ sample_flows.csv already exists")
        return

    rng = np.random.RandomState(42)
    classes = ['Normal','DoS','Probe','R2L','U2R']
    weights = [0.6, 0.2, 0.1, 0.07, 0.03]
    labels  = rng.choice(classes, size=n, p=weights)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"feat_{i:02d}" for i in range(41)] + ["label"]
        writer.writerow(header)
        for lbl in labels:
            row = list(rng.rand(41).round(4)) + [lbl]
            writer.writerow(row)

    print(f"  Generated sample_flows.csv ({n} rows)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['nslkdd','cicids','all'], default='nslkdd')
    parser.add_argument('--data-dir', default='data')
    args = parser.parse_args()

    if args.dataset in ('nslkdd','all'):
        download_nslkdd(args.data_dir)
    if args.dataset in ('cicids','all'):
        download_cicids()

    generate_sample_flows(args.data_dir)
    print("✅ Dataset setup complete.")


if __name__ == "__main__":
    main()
