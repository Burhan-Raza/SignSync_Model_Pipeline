# ASL-20 Conformer with face/body points: final model

Final model of this repository. 20 ASL signs, tested on 4 signers the model never saw in training.
**Test accuracy 85.3%** (1,080 of 1,266 recordings).

Files here: `sign_conformer.onnx` + `sign_conformer.onnx.data` (model), `labels.json` (class names),
`conformer_best.pt` (PyTorch weights), `conformer_results_asl20_body.txt` (training log),
`input_aspect.txt` (image shape of the training videos, used by `live_predict.py`).

## Input
Per frame, from MediaPipe Holistic (`body.py`):
- both hands, 21 points each (126 numbers, `[left | right]`, missing hand = 0);
- 13 face/body points: nose, both eyes, both ears, both mouth corners, both shoulders, both elbows (pose) and
  forehead and chin (face mesh).

Model features per frame (172): each hand's shape (wrist-relative, scaled by hand size), wrist and 4 fingertips
of each hand relative to the nose, and the 13 face/body points relative to the nose. Positions are divided by the
signer's shoulder width (x) and nose-to-shoulder height (y), so they say where the hand is on the face or body,
independent of camera distance. Each sign is trimmed to the frames with hands and resampled to 60 frames.

## Model and training
`train_conformer.py --data asl20_body.npz --features body --epochs 80`: 3 Conformer blocks (feed-forward, 4-head
self-attention, depthwise convolution with kernel 15), width 128, 1,182,996 parameters. Augmentation: zoom, shift,
jitter, speed change, cut ends, dropped hand frames, random left/right mirror (hands and body together).
Dropout 0.2, AdamW (lr 5e-4, weight decay 0.01, 5 warm-up epochs, cosine), seed 0, CPU, about 66 minutes.
Early stopping ended training at epoch 49; the saved model is from epoch 29 (best validation accuracy).

## Results (same 1,266 test recordings for all models)

| Model | Input | Train | Val | **Test** | Stress test |
|---|---|---|---|---|---|
| GRU (comparison, not in this repository) | hands | 98.6% | 92.0% | 83.2% | 82.9% |
| Conformer (comparison, not in this repository) | hands | 98.5% | 92.5% | 85.2% | 83.8% |
| **Conformer (this model)** | **hands + face/body** | 97.5% | **96.8%** | **85.3%** | 84.0% |

Per test signer: 62590 98%, 32319 86%, 49445 86%, 4718 63% (hands-only Conformer: 96%, 90%, 88%, 57%).

Per sign (test): bye 96%, hello 97%, home 95%, time 95%, sick 92%, no 91%, water 91%, where 91%, callonphone 89%,
thankyou 88%, sleep 87%, drink 86%, hungry 85%, yes 84%, see 80%, happy 77%, wait 74%, please 72%, fine 69%, sad 68%.

Most common mistakes: sad -> wait, please -> hungry, sad -> sleep, yes -> time, fine -> time, happy -> thankyou.

Compared with the hands-only Conformer, validation accuracy rose from 92.5% to 96.8% but test accuracy stayed
the same (85.3% vs 85.2%; 59 test recordings right only with this model, 57 only with the other). Face-located
signs improved for some signs (bye, thankyou) and for the hardest test signer (4718: 57% to 63%).

## Live use
```bash
cd asl20_conformer_face
python ../live_predict.py --hold 2
```
- Keep your head and both shoulders in the frame: positions are measured from them.
- `input_aspect.txt` (0.88) is the width/height of the Kaggle videos, estimated from hand geometry. The live
  script rescales webcam x to that shape; on test signs viewed as through a 4:3 webcam this restores accuracy from
  82.9% to 85.3%. Turn it off with `--no-aspect-fix`.
- Answers below 50% confidence are not shown (`--threshold 0` shows all).

## Limits
- One training run (seed 0); seeds moved test accuracy by about 1 point in earlier runs.
- Validation has only 2 signers, so the 96.8% does not carry over to the test signers.
- The test signers were also used to compare earlier models, so 85.3% is not a completely untouched estimate.
- In the Kaggle data `sick` and the other signs are one-handed; two-handed versions are not recognized well.
- Accuracy on a live webcam was not measured.
