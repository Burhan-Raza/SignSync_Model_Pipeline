"""Download ONLY the ASL signs you need from Kaggle asl-signs (instead of the full 57 GB).

Steps it does:
  1. downloads train.csv (the list of all sequences: path, participant_id, sequence_id, sign)
  2. picks your signs, at most --max-per-sign sequences each, spread over all participants
  3. downloads just those .parquet files into  <out>/train_landmark_files/<participant>/<seq>.parquet
  4. (optional, --convert) runs convert_asl.py so you directly get asl_data/ in PSL format

One-time setup:
  pip install kaggle pandas pyarrow
  Kaggle -> Settings -> API -> "Create New Token" -> put kaggle.json in
      Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json     Linux/Mac/Colab: ~/.kaggle/kaggle.json
  Open https://www.kaggle.com/competitions/asl-signs/data and accept the rules (else: 403 error).

Usage:
  python download_asl.py --signs yes where hungry sick time callonphone home see --max-per-sign 120 --convert
  python download_asl.py --list            # only download train.csv and show all 250 signs
Safe to re-run: files that are already downloaded are skipped.
"""
import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

COMP = "asl-signs"


def kaggle_file(remote, dest_dir):
    """Download one competition file into dest_dir (Kaggle sometimes zips it - unzip then)."""
    os.makedirs(dest_dir, exist_ok=True)
    name = os.path.basename(remote)
    target = os.path.join(dest_dir, name)
    if os.path.exists(target) and os.path.getsize(target) > 0 and not zipfile.is_zipfile(target):
        return "skip"
    if os.path.exists(target):
        os.remove(target)          # empty or still-zipped leftover from an earlier run
    exe =shutil.which("kaggle", path=os.path.dirname(sys.executable)) or shutil.which("kaggle")   # venv first
    cmd = [exe] if exe else [sys.executable, "-m", "kaggle"]
    r = subprocess.run(cmd + ["competitions", "download", "-c", COMP, "-f", remote, "-p", dest_dir, "-q"],
                       capture_output=True, text=True)
    z = target + ".zip"
    if os.path.exists(target) and zipfile.is_zipfile(target):
        os.replace(target, z)      # Kaggle can also send a zip under the plain file name
    if os.path.exists(z):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest_dir)
        os.remove(z)
    if not os.path.exists(target):
        msg = (r.stderr or r.stdout).strip()
        raise RuntimeError(f"{remote}: {msg or 'download failed'}")
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="asl-signs", help="local folder (same layout as the Kaggle data)")
    ap.add_argument("--signs", nargs="*", help="ASL signs to download")
    ap.add_argument("--max-per-sign", type=int, default=120)
    ap.add_argument("--workers", type=int, default=6, help="parallel downloads")
    ap.add_argument("--list", action="store_true", help="download train.csv and print all signs")
    ap.add_argument("--convert", action="store_true", help="run convert_asl.py afterwards")
    ap.add_argument("--asl-out", default="asl_data", help="output folder for --convert")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import pandas as pd
    try:
        kaggle_file("train.csv", a.out)
    except RuntimeError as e:
        sys.exit(f"Could not download train.csv.\n{e}\n\nCheck: kaggle.json is in place, and you accepted the "
                 f"rules at https://www.kaggle.com/competitions/{COMP}/data")
    meta = pd.read_csv(os.path.join(a.out, "train.csv"))
    counts = meta["sign"].value_counts().sort_index()
    if a.list or not a.signs:
        print(f"{len(counts)} signs, {len(meta)} sequences, {meta['participant_id'].nunique()} participants")
        print(", ".join(f"{s}({n})" for s, n in counts.items()))
        if not a.signs:
            print("\nNow run again with --signs <sign1> <sign2> ...")
        return

    missing = [s for s in a.signs if s not in counts.index]
    if missing:
        sys.exit(f"Not in asl-signs (check spelling with --list): {missing}")
    sel = meta[meta["sign"].isin(a.signs)].sample(frac=1, random_state=a.seed)
    sel = sel.assign(r=sel.groupby(["sign", "participant_id"]).cumcount())
    sel = sel.sort_values(["sign", "r"]).groupby("sign").head(a.max_per_sign)
    print(f"downloading {len(sel)} sequences: " + ", ".join(f"{s} {n}" for s, n in sel['sign'].value_counts().sort_index().items()))

    done = fail = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(kaggle_file, p, os.path.join(a.out, os.path.dirname(p))): p for p in sel["path"]}
        for f in as_completed(futs):
            try:
                f.result(); done += 1
            except RuntimeError as e:
                fail += 1
                if fail <= 3:
                    print("  failed:", e)
            if (done + fail) % 50 == 0:
                print(f"  {done + fail}/{len(sel)}")
    print(f"downloaded {done}, failed {fail}" + ("  (re-run the same command to retry the failed ones)" if fail else ""))

    # keep only the rows we actually have, so convert_asl.py doesn't look for missing files
    have = sel[[os.path.exists(os.path.join(a.out, p)) for p in sel["path"]]]
    have.drop(columns="r").to_csv(os.path.join(a.out, "train_selected.csv"), index=False)
    print(f"list of downloaded sequences -> {os.path.join(a.out, 'train_selected.csv')}")

    if a.convert:
        here = os.path.dirname(os.path.abspath(__file__))
        subprocess.run([sys.executable, os.path.join(here, "convert_asl.py"), "--kaggle", a.out,
                        "--csv", "train_selected.csv", "--out", a.asl_out], check=True)


if __name__ == "__main__":
    main()
