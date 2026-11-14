# Model weights

The weight files in this folder are **not committed**. They are downloaded and SHA-256
verified by `scripts/download_models.py` (run automatically by `setup.bat`).
`scripts/check_models.py` proves they load and run.

| File | Used for | Source | Licence |
|---|---|---|---|
| `face_landmarker.task` (3.8 MB) | Face detection + 478 landmarks (alignment, quality checks, parallax liveness) | Google MediaPipe: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task` | Apache-2.0 |
| `w600k_r50.onnx` (174 MB) | Face recognition: ArcFace ResNet-50 trained on WebFace600K, 512-d embedding, input 112x112 RGB, (x-127.5)/127.5 | InsightFace `buffalo_l` model pack, official release `https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip`. Downloaded by default from the single-file mirror `https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx`, which is **byte-identical** to the file in the official zip (same SHA-256); the official zip is the fallback. | InsightFace pretrained models: **non-commercial research use only** (InsightFace code itself is MIT). Fine for this university research project; a commercial deployment would need a different recognition model or a licence from InsightFace. |
| `depth_anything_v2_small.onnx` (99 MB) | Monocular relative depth of the face crop (depth liveness). Output is relative *inverse* depth: larger = closer. Input: RGB/255, ImageNet mean/std, side a multiple of 14. | ONNX export of Depth Anything V2 **Small** by onnx-community (Hugging Face): `https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx` | Apache-2.0 (Small only; Base/Large/Giant are CC-BY-NC-4.0) |
| `MiniFASNetV2.onnx` (1.7 MB) | Texture anti-spoofing, crop scale 2.7 | ONNX export of the Silent-Face-Anti-Spoofing (minivision-ai) weights, from `https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx` | Apache-2.0 (both upstream and the export repo) |
| `MiniFASNetV1SE.onnx` (1.7 MB) | Texture anti-spoofing, crop scale 4.0 | same repo: `.../weights/MiniFASNetV1SE.onnx` | Apache-2.0 |

## Notes

- **MiniFASNet** input: a face crop enlarged around the face box by the given scale, resized
  to 80x80, **BGR, raw 0-255 values (not divided by 255)**. Output: 3 logits; after softmax,
  class 1 = real, classes 0 and 2 = spoof types. As in the original project, the two models
  are used together and their probabilities averaged.
  The ONNX files are a third-party conversion, so their correctness was checked against the
  original project's labelled sample images (`images/sample/` in minivision-ai/Silent-Face-Anti-Spoofing):
  real sample P(real) = 1.000; the two fake samples 0.065 and 0.002, i.e. the expected behaviour.
  Crop details that matter (implemented in `app/vision/liveness_texture.py`): the face box is a
  square around the MediaPipe mesh (side = longer edge), which matches the box of the original
  project's RetinaFace detector to within a few percent on test images; the enlarged box is
  shifted inside the frame rather than clipped; and black letterbox bars must be removed from the
  frame first - with the bars present, a real face scored P(real) ~0.00, without them ~0.3-0.5
  (on a very dark frame). The model is also sensitive to darkness/noise and to the face filling
  the frame, which the pipeline's quality checks guard against.
- **ArcFace** alignment: 5 points (eye centres, nose tip, mouth corners) taken from the
  MediaPipe landmarks (indices 468, 473, 1, 61, 291) are mapped to the standard ArcFace
  112x112 template with a similarity transform (`cv2.estimateAffinePartial2D`).
- Pinned SHA-256 hashes are in `scripts/download_models.py`.
