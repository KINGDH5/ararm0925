# pages/01_시나리오1.py
import os, runpy, streamlit as st

# 멀티페이지에서 set_page_config 중복 경고 피하고 싶으면 환경변수 하나 심어줌
os.environ["IS_MULTIPAGE"] = "1"

# 올바른 코드 (새 경로)
SCRIPT = r"C:\Users\권도혁\Desktop\시나리오1,2\app.py"

# 실행
runpy.run_path(SCRIPT, run_name="__main__")
