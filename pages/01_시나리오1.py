import runpy
from pathlib import Path

# 프로젝트 루트의 app.py 실행
SCRIPT = Path(__file__).resolve().parents[1] / "app.py"
runpy.run_path(str(SCRIPT), run_name="__main__")
