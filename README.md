# One model, two sign languages (8 PSL + 8 ASL = 16 classes)
```bash
python prepare.py --root "$ROOT" --out data8.npz --eight
python download_asl.py --signs yes where hungry sick time callonphone home see --max-per-sign 120 --convert
python prepare_asl.py --asl asl_data --psl data8.npz --out psl_asl.npz
python baseline_sklearn.py psl_asl.npz --train-on-all     # saves baseline_model.joblib
python train_gru.py --data psl_asl.npz --epochs 60        # saves gru_best.pt, labels.json, sign_gru.onnx
python live_baseline.py --lang psl                        # psl | asl | both (default both)
python predict.py --seq-dir some/seq --lang asl
```
Both scripts report test accuracy for all classes and separately for `psl_*` / `asl_*`. `--lang` zeroes the other
language's probabilities and renormalizes; words are shown without the prefix.
`train_gru.py` now trains from `X_raw` (`augment_raw` + random `mirror`, then `trim_active` + `normalize`, like `baseline_sklearn.py`).

---

# ASL (Kaggle asl-signs) in the same format

`convert_asl.py` turns Kaggle "Google - Isolated Sign Language Recognition" parquet files into the PSL layout
(`<word>/<seq>/0..59.npy`, 126 numbers per frame, [left | right], missing hand = 0; face and pose are dropped).
`prepare_asl.py` makes an .npz that `baseline_sklearn.py` / `train_gru.py` read unchanged, with a
**signer-independent split** (whole participants held out for test). Add `--psl data8.npz` to train one model on both;
labels become `psl_<word>` / `asl_<word>`.
**Don't download the full 57 GB.** `download_asl.py` gets only the signs you need (~960 small files):
```bash
pip install kaggle pandas pyarrow          # + kaggle.json token, + accept rules on the competition Data page
python download_asl.py --list
python download_asl.py --signs yes where hungry sick time callonphone home see --max-per-sign 120 --convert
```
Or by hand, if you already have the full data:
```bash
pip install pandas pyarrow
python convert_asl.py --kaggle asl-signs/ --list                                   # see the 250 signs
python convert_asl.py --kaggle asl-signs/ --out asl_data --signs hello water bye thankyou --max-per-sign 120
python prepare_asl.py --asl asl_data --out asl.npz                                 # ASL alone
python prepare_asl.py --asl asl_data --psl data8.npz --out psl_asl.npz             # PSL + ASL together
python baseline_sklearn.py psl_asl.npz
```

---

# Fix for live webcam (8 app signs) - read this first

**Problem:** on the dataset's webcam test set, excuseme / have_a_good_day / goodbye are already 93-100%,
but live in front of your own camera they turn into welldone / goodmorning. The model was fine on clean
recordings and broke on live input, because the live input differed in four ways:
1. selfie/mirror flip (`cv2.flip`) swaps the left/right hand slots and mirrors x;
2. the live window doesn't start/end with the sign (idle frames, a sign that gets cut off);
3. signing speed is different from the 60-frame recordings;
4. camera framing / distance is different (absolute wrist position is a feature).

**What changed**
- `common.py`: `trim_active` (cut empty start/end + resample to 60), `fill_gaps`, `mirror`, `augment_raw`, `EIGHT_SIGNS`.
- `prepare.py`: trims every sequence, saves `X_raw` too, `--eight` shortcut, `--extra-root` for your own recordings.
- `baseline_sklearn.py`: trains on mirrored + shifted/zoomed/speed-changed copies, better features,
  prints a **stress (live-like) test** next to the clean test, saves `baseline_model.joblib`.
- `live_baseline.py` (new): webcam / video / folder prediction using the exact same preprocessing,
  "not sure" threshold, and `--collect` to record your own samples.

**Run**
```bash
python prepare.py --root "$ROOT" --out data8.npz --eight
python baseline_sklearn.py data8.npz                       # compare: add --aug 0 for the old behaviour
python live_baseline.py                                    # SPACE = record one sign, q = quit
```
Live tips: hands down (out of frame) between signs, perform the whole sign, normal speed.

**Training data:** every model here (baseline and GRU) is trained ONLY on PakistanSignLanguageDatasetV2. No manual recording is needed.
`data8.npz` and `baseline_model.joblib` in this folder are already built from the dataset (all 3 cameras).

**Optional (not needed, not used by default):** record your own samples and add them to training:
```bash
python live_baseline.py --collect excuseme --save-dir my_data      # SPACE x 20, repeat for each word
python prepare.py --root "$ROOT" --out data8.npz --eight --extra-root my_data
python baseline_sklearn.py data8.npz --train-on-all
```

---

# SignSync AI — PSL recognition pipeline (Steps 2–5)

## What's in your archive
`PakistanSignLanguageDatasetV2` doesn't contain videos. It holds **landmarks that were already extracted with MediaPipe**:

```
<source>/<word>/<sequence>/<frame>.npy     source = laptop_data | mobile_data | webcam_data
each frame = 126 numbers = 2 hands × 21 points × (x, y, z)   [left hand | right hand], missing hand = 0
each sequence = 60 frames
```
- 3 sources, about 6,900 usable sequences in total, 71 words (the `links_to_words_final.txt` file maps each word to its psl.org.pk dictionary page)
- Junk folders to ignore: `s`, `test_word`, `test2`. A few sequences are shorter than 60 frames.
- 10 words don't appear in the webcam source and get dropped from the cross-camera test: bird, come, doctor, give, help, hungry, need, want, where, who. `nothing` also gets dropped because it has too few clean samples.

## Run order
```bash
# Python 3.10/3.11 recommended (needed for mediapipe 0.10.14)
pip install -r requirements.txt
ROOT="<path>/PakistanSignLanguageDatasetV2/PakistanSignLanguageDatasetV2"

python prepare.py --root "$ROOT" --out pilot5.npz --words hello water bring fan you   # Step 2 pilot
python prepare.py --root "$ROOT" --out data_prepared.npz                              # Step 4 full
python baseline_sklearn.py data_prepared.npz                                          # baseline
python train_gru.py --data data_prepared.npz                                          # Step 5 (GRU, use Colab GPU if slow)
python predict.py --video my_sign.mp4                                                 # inference
python extract_landmarks.py --video-dir videos/ --out new_data/                       # Step 3 for new PSL/ASL videos
```

## Results so far (baseline, ExtraTrees on summary features)
| Split | Test accuracy |
|---|---|
| 5-sign pilot, cross-camera | 100% |
| 61 words, **train laptop+mobile → test webcam** | **90.7%** |
| 61 words, random 70/15/15 | 99.4% |

**Important for your thesis:** the dataset has no signer IDs, so a true signer-wise split isn't possible.
The random split gives 99%, but near-identical recordings land in both train and test, so that number
is inflated. Report the **cross-camera (webcam held-out)** number as your main result. The weakest words
are toothpaste/toothbrush (very similar signs), garden, all, you, and shower. These are good targets for the GRU and augmentation.

## Files
- `common.py`: loading (frames sorted numerically, so 0,1,2…10 rather than 0,1,10), normalization (wrist-relative + scale, plus the wrist position in the image)
- `prepare.py`: cleaning, labels, splits → `.npz`
- `baseline_sklearn.py`: quick classical baseline
- `train_gru.py`: bidirectional GRU + augmentation, early stopping, exports `sign_gru.onnx` + `labels.json`
- `predict.py`: video/sequence → top-3 words (this is the function your FastAPI endpoint should call)
- `live_predict.py`: webcam prediction; smooths recent outputs and waits for a stable two-of-three vote before showing a sign
- `extract_landmarks.py`: MediaPipe Holistic → the dataset's exact format (60 × 126). Use it for new PSL dictionary videos, ASL, or webcam recordings.

## Notes / checks
- **Hand order:** the dataset matches Holistic's `[left_hand, right_hand]` (no mirror flip). Right-handed signs land in slot 2.
  Check one of your own videos with `--verify-against <dataset seq folder>`. If the slots come out reversed, swap them.
- If your app shows a mirrored selfie preview, still send the **un-mirrored** frames to MediaPipe.
- Untested here: `train_gru.py`, `predict.py` and real MediaPipe extraction (PyTorch/MediaPipe couldn't be installed in my environment).
  The data pipeline and baseline actually ran, and the extraction output format was checked against the dataset.
