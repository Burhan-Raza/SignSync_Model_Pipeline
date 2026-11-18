# SignSync: ASL sign recognition (20 signs)

Recognizes 20 American Sign Language (ASL) signs from a webcam. MediaPipe Holistic gives hand, face and body
landmarks for every frame, and a Conformer network classifies each sign.

## The 20 ASL signs

| | | | | |
|---|---|---|---|---|
| bye | callonphone | drink | fine | happy |
| hello | home | hungry | no | please |
| sad | see | sick | sleep | thankyou |
| time | wait | water | where | yes |

## Model used

**Conformer with hand + face/body points** (folder [`asl20_conformer_face/`](asl20_conformer_face/RESULTS.md)).

- **Input per frame:** both hands (2 x 21 points) and 13 face/body points: nose, eyes, ears, mouth corners,
  shoulders, elbows, forehead and chin. Hand positions are measured relative to the face and shoulders, so the
  model can tell where on the face or body a sign is made.
- **Network:** 3 Conformer blocks (feed-forward, 4-head self-attention, depthwise convolution), width 128,
  1.18 million parameters, 60 frames per sign. The idea comes from Haque et al., "A Signer-Invariant Conformer and
  Multi-Scale Fusion Transformer for Continuous Sign Language Recognition" (ICCV 2025 Workshops); here a small
  Conformer classifies single signs.
- **Data:** Kaggle "Google - Isolated Sign Language Recognition" (asl-signs): all 7,597 recordings of these 20
  signs, 6,679 kept after cleaning, from 21 signers.
- **Split by signer:** 15 signers train (4,734 recordings), 2 val (679), 4 test (1,266). Nobody appears in two
  splits, so the test measures accuracy on new people.

## Results (4 test signers never seen in training)

| Model | Input | Test accuracy |
|---|---|---|
| GRU | hands | 83.2% |
| Conformer | hands | 85.2% |
| **Conformer (final)** | **hands + face/body** | **85.3%** |

Per signer, the final model scores 98%, 86%, 86% and 63%. Weakest signs: sad, fine, please, wait, happy.
Only the final model is in this repository; the GRU and hands-only Conformer were trained on the same data and
split for comparison. Details, per-sign numbers and limits: [asl20_conformer_face/RESULTS.md](asl20_conformer_face/RESULTS.md).

## Run it on a webcam

```bash
pip install -r requirements.txt          # Python 3.10 (mediapipe 0.10.14)
cd asl20_conformer_face
python ../live_predict.py --hold 2
```

- Raise your hands and sign; the word shows while your hands are in the frame and clears when they leave.
  `--hold 2` keeps it on screen for 2 seconds.
- Keep your head and both shoulders in the frame.
- Answers below 50% confidence are not shown; `--threshold 0` shows everything.
- The webcam image is rescaled to the shape of the training videos (`input_aspect.txt`); `--no-aspect-fix`
  turns this off.
- `S` saves the last sign and prints the top 3 answers, `Q` quits.

`live_predict.py` and `predict.py` load `sign_conformer.onnx` from the folder you run them in (otherwise
`sign_gru.onnx`); `--model` picks a file.

## Rebuild the data and train

```bash
pip install kaggle pandas pyarrow        # kaggle.json in ~/.kaggle, competition rules accepted
python download_asl.py --signs bye callonphone drink fine happy hello home hungry no please sad see sick sleep thankyou time wait water where yes --max-per-sign 1000
python prepare_asl_body.py --out asl20_body.npz
mkdir asl20_conformer_face && cd asl20_conformer_face
python ../train_conformer.py --data ../asl20_body.npz --features body --epochs 80
```
Kaggle limits per-file downloads (HTTP 429 after a few hundred files). Re-running `download_asl.py` continues
where it stopped; for this project the files were zipped by a private Kaggle notebook and downloaded as one file.

## Files

| File | What it does |
|---|---|
| `live_predict.py` | Webcam recognition (one answer per sign) |
| `predict.py` | Predict one recorded sign (`--seq-dir`) |
| `body.py` | Face/body points: reading, trimming, mirroring, augmentation, features |
| `common.py` | Hand landmark helpers (normalize, trim, mirror, augmentation) |
| `train_conformer.py` | Train the Conformer (`--features hands` or `body`) |
| `train_gru.py` | Train the GRU (comparison model) |
| `download_asl.py`, `convert_asl.py`, `prepare_asl.py` | Download and convert Kaggle asl-signs (hands only) |
| `prepare_asl_body.py` | Build the hands + face/body dataset |
| `extract_landmarks.py` | MediaPipe Holistic landmark extraction |
| `live_baseline.py`, `baseline_sklearn.py` | Classical (scikit-learn) baseline and sample recording (`--collect`) |
| `asl20_conformer_face/` | **Final model** |
