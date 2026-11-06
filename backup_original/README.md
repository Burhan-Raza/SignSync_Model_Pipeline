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
