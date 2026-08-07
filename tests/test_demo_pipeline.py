from pathlib import Path

from visionguard.config import ROOT
from visionguard.demo import run_demo


def test_demo_pipeline_processes_video(tmp_path: Path):
    result = run_demo("datasets/demo/demo_scene.mp4", str(tmp_path / "out.mp4"), max_frames=200)
    assert result["frames"] == 200
    assert result["events"] >= 1
    assert Path(result["output"]).exists()
