.PHONY: install install-full test lint demo serve data

install:
	python -m pip install -r requirements-demo.txt

install-full:
	python -m pip install -r requirements.txt

test:
	python -m pytest

lint:
	python -m ruff check .

data:
	python scripts/generate_demo_assets.py

demo:
	python -m visionguard demo --source datasets/demo/demo_scene.mp4

serve:
	python -m visionguard serve
