# Home.py — GitHub/Streamlit 배포용 홈 대시보드
import os
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="메인 페이지", page_icon="👋", layout="wide")
st.sidebar.success("탐색할 페이지를 선택하세요.")

st.write("# 멀티페이지 홈 👋")
st.markdown("왼쪽 **pages/** 에서 시나리오1/2를 선택하세요.")

# ─────────────────────────────────────────
# 빠른 이동 (Streamlit 1.32+)
# ─────────────────────────────────────────
st.subheader("빠른 이동")
cols = st.columns(2)
with cols[0]:
    st.page_link("pages/01_시나리오1.py", label="➡️ 시나리오 1 열기", icon=":material/rocket_launch:")
with cols[1]:
    st.page_link("pages/02_시나리오2.py", label="➡️ 시나리오 2 열기", icon=":material/auto_awesome:")

st.divider()

# ─────────────────────────────────────────
# 환경/데이터/시크릿 간단 진단
# ─────────────────────────────────────────
st.subheader("상태 점검")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "renamed_data.csv"

# 데이터 파일 존재 여부
data_ok = DATA_PATH.exists()
st.write("**데이터 파일**:", "✅ 존재" if data_ok else "❌ 없음", f"({DATA_PATH})")

# 시크릿 존재 여부 (Vertex 사용 시)
proj = st.secrets.get("PROJECT_ID", None)
endpoint = st.secrets.get("ENDPOINT_ID", None)
region = st.secrets.get("REGION", "us-central1")
secrets_ok = bool(proj and endpoint)
st.write("**Vertex 시크릿**:", "✅ 설정됨" if secrets_ok else "⚠️ 미설정",
         f"(PROJECT_ID={proj or '-'}, ENDPOINT_ID={endpoint or '-'}, REGION={region})")

with st.expander("문제 해결 팁"):
    st.markdown(
        """
        - `data/renamed_data.csv`가 없으면 **파일 업로드 모드**로 실행하거나, 작은 샘플을 `data/`에 넣어주세요.  
        - Vertex 감지를 쓰려면 **Streamlit Cloud → Settings → Secrets**에 아래 키를 넣으세요:
            - `PROJECT_ID`, `REGION`, `ENDPOINT_ID`, `GOOGLE_APPLICATION_CREDENTIALS_B64`(선택)
        - 로컬 개발 시 `.streamlit/secrets.toml`(gitignore 처리)에 같은 키를 넣으면 됩니다.
        """
    )
