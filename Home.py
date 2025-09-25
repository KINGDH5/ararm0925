# -*- coding: utf-8 -*-
"""
Home.py - Streamlit 메인
- 깃허브/Streamlit Cloud 기준: 경로 자동 탐색 + 안전 로더 + 폴백 학습
- 모델(.joblib/.pkl) 또는 데이터가 없어도 초기화 실패 없이 동작
"""
import os, pickle, traceback
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI 기반 LoL 아이템 빌드 추천", layout="wide")
st.title("AI 기반 LoL 아이템 빌드 추천")

BASE_DIR = Path(__file__).resolve().parent

# ── Debug 패널: Cloud에서도 경로/존재 여부 즉시 확인 ─────────────────
with st.sidebar:
    st.markdown("### 🧭 Debug")
    st.write("CWD:", os.getcwd())
    st.write("BASE_DIR:", str(BASE_DIR))

def file_status(label: str, p: Path):
    st.sidebar.write(f"{label}: {p} ", "✅" if (p and p.exists()) else "❌")

def try_or_alert(msg, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        st.error(f"🚨 {msg}")
        st.exception(e)
        st.code("".join(traceback.format_exc()))
        st.stop()

# ── 리포 구조에 맞춘 후보 경로(존재하는 첫 파일 채택) ────────────────
MODEL_CANDIDATES = [
    BASE_DIR / "lgbm_model_tuned.joblib",           # 리포에 실제 존재
    BASE_DIR / "models" / "scenario1_model.pkl",    # 과거 형식(있으면 사용)
]
DATA_CANDIDATES = [
    BASE_DIR / "renamed_data_sample.csv",           # 리포에 실제 존재
    BASE_DIR / "renamed_data.csv",                  # 과거 형식(있으면 사용)
]

def first_exist(paths):
    for p in paths:
        file_status("CHECK", p)
        if p.exists():
            return p
    return None

MODEL_PATH = first_exist(MODEL_CANDIDATES)
DATA_PATH  = first_exist(DATA_CANDIDATES)

st.sidebar.write("MODEL_PATH:", MODEL_PATH if MODEL_PATH else "None")
st.sidebar.write("DATA_PATH:",  DATA_PATH if DATA_PATH else "None")

# ── 캐시 로더 ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource(show_spinner=False)
def load_model_pickle_or_joblib(path: Path):
    p = str(path).lower()
    if p.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        import joblib
        return joblib.load(path)

# ── 데이터 로드 ──────────────────────────────────────────────────────
if DATA_PATH is None:
    st.error("데이터 파일이 없습니다. (renamed_data_sample.csv 또는 renamed_data.csv)")
    st.stop()
df = try_or_alert("데이터 로드 실패", load_df, DATA_PATH)

# ── 모델 로드(없으면 폴백 학습) ─────────────────────────────────────
if MODEL_PATH is None:
    from ml import train_models  # 리포의 ml.py
    st.warning("모델 파일이 없어 임시 학습을 수행합니다. (첫 실행만 지연될 수 있음)")
    models = try_or_alert("임시 학습 실패", train_models, df)
    st.session_state["models"] = models
    st.success("✅ 임시 학습 완료: 세션에 모델을 보관했습니다.")
else:
    model_or_bundle = try_or_alert("모델 로드 실패", load_model_pickle_or_joblib, MODEL_PATH)
    st.session_state["models"] = model_or_bundle
    st.success(f"✅ 모델 로드 완료: {MODEL_PATH.name}")

# ── 상태 카드 & 사용 안내 ───────────────────────────────────────────
c1, c2, c3 = st.columns(3)
with c1: st.metric("데이터 로우 수", f"{len(df):,}")
with c2: st.metric("데이터 경로", DATA_PATH.name)
with c3: st.metric("모델 상태", "세션 로드됨")

st.info("왼쪽 **Pages**에서 ① 시나리오1 / ② 시나리오2 페이지로 이동해 기능을 사용하세요.")

# 필요한 경우: st.session_state["models"] 구조를 확인하려면 아래 주석 해제
# st.write("session keys:", list(st.session_state.keys()))
# st.write(st.session_state.get("models"))
