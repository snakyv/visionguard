# Verification report

The repository is designed so core application behavior can be verified without downloading ML weights.

## Verified in the build environment

- Python source compilation with `compileall`.
- Editable package build with PEP 517 using `--no-build-isolation`.
- Five automated pytest tests.
- Synthetic detector recognition of all bundled classes.
- Identity persistence in the deterministic tracker.
- Zone enter/exit event generation.
- FastAPI health and camera endpoints.
- End-to-end processing of the bundled MP4 through detection, tracking, analytics and video rendering.
- Dataset validation for all bundled images and YOLO labels.
- REST start/stop lifecycle for the demo camera worker.
- Runtime metrics and event API population while the camera is running.

## Build smoke result

A 220-frame demo run produced three analytics events and a valid annotated MP4 in the build environment. Performance numbers from this synthetic backend are not meaningful ML inference benchmarks and must not be used as portfolio model-performance claims.

## External-stack verification boundary

The build environment has no outbound package-network access, so Ultralytics, Roboflow `trackers`, OpenVINO and NNCF could not be installed during artifact creation. Their integration code and version ranges were checked against the current official APIs, but training, real ByteTrack execution and OpenVINO export must be re-run locally after `pip install -r requirements.txt`.

Run the complete local verification with:

```powershell
pytest
python training/dataset_validator.py --data datasets/demo/dataset.yaml
python -m visionguard demo --source datasets/demo/demo_scene.mp4
```
