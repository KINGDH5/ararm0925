# pages/02_시나리오2.py — GitHub/Streamlit 배포용
import os
import sys
import runpy
from pathlib import Path

# === 새 위치 (레포 루트 기준) ===
BASE2 = Path(__file__).resolve().parent.parent / "시나리오2"
APP2  = BASE2 / "app.py"

# 1) 이전에 로드된 동일 모듈 캐시 제거
for mod in ["item_recommender", "rune_champion"]:
    if mod in sys.modules:
        sys.modules.pop(mod, None)

# 2) 시나리오2 폴더를 sys.path 최상단에 추가
if str(BASE2) in sys.path:
    sys.path.remove(str(BASE2))
sys.path.insert(0, str(BASE2))

# 3) CWD를 시나리오2로 바꾸고 실행 (상대경로 보호)
_prev = os.getcwd()
os.chdir(BASE2)
try:
    runpy.run_path(str(APP2), run_name="__main__")
finally:
    os.chdir(_prev)
