# app.py (시나리오2/kdh용 · 예상 승률 제거판, GitHub/Streamlit 배포용)
# -*- coding: utf-8 -*-
import os
import io
import sys
import importlib
from pathlib import Path

import streamlit as st
from PIL import Image
from item_recommender import (
    initialize_recommender,
    get_all_build_recommendations,
    get_cc_df,
)

# === 고정 경로 제거: 현재 파일 기준으로 BASE_DIR 설정 ===
BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# rune_champion.py 가 같은 폴더(또는 하위 폴더)에 있을 때 import 보장
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

st.set_page_config(page_title="AI 기반 LoL 아이템 빌드 추천",
                   page_icon="🤖", layout="wide")
st.title("AI 기반 LoL 아이템 빌드 추천")
st.markdown("---")

# 0) 추천 엔진 초기화
if not initialize_recommender():
    st.error("추천 시스템 초기화 실패 (모델/데이터 경로를 확인하세요)")
    st.stop()


# ──────────────────────────────────────────────
# 적 조합 요약 + 추천 이유
# ──────────────────────────────────────────────
def summarize_enemy(enemy_team):
    num_ad = num_ap = num_tanks = num_support = num_cc = 0
    cc_df = get_cc_df()

    for champ_name, _, role_name in enemy_team:
        role_name = str(role_name or "")
        if role_name.startswith("AD"):       num_ad += 1
        elif role_name.startswith("AP"):     num_ap += 1
        elif role_name.startswith("탱커"):    num_tanks += 1
        elif role_name.startswith("서포터"):  num_support += 1

        # CC 계산
        row = cc_df[cc_df["name"] == champ_name]
        if not row.empty:
            try:
                num_cc += int(row.iloc[0]["CCcount"])
            except Exception:
                pass

    return num_ad, num_ap, num_tanks, num_support, num_cc


def render_reasons(num_ad, num_ap, num_tanks, num_support, num_cc):
    reasons = []
    reasons.append(
        f"적 조합 요약: AD {num_ad}명 / AP {num_ap}명 / 탱커 {num_tanks}명 / 서포터 {num_support}명 / 총 CC {num_cc}개"
    )

    # AD/AP 비교
    if num_ap > num_ad:
        reasons.append("상대 조합에 **AP 챔피언 비중이 높아** 마법 저항력 아이템이 유리합니다.")
    elif num_ad > num_ap:
        reasons.append("상대 조합에 **AD 챔피언 비중이 높아** 방어력 아이템이 유리합니다.")
    else:
        reasons.append("상대 조합의 **AD/AP 비중이 비슷**하여 상황에 맞게 방어/마저를 혼합하는 편이 좋습니다.")

    # CC 체크
    if num_cc >= 3:
        reasons.append("상대 조합에 **CC가 많아 강인함/군중제어 대응** 아이템이 중요합니다.")

    return reasons


# ──────────────────────────────────────────────
# 1) 스크린샷 업로드
# ──────────────────────────────────────────────
st.header("1. 정보 입력")
uploaded = st.file_uploader("게임 로딩 화면 스크린샷 업로드", type=["png", "jpg", "jpeg"])
temp_path, champs10 = None, []

if uploaded:
    temp_path = TEMP_DIR / uploaded.name
    with open(temp_path, "wb") as f:
        f.write(uploaded.getvalue())
    st.image(Image.open(io.BytesIO(uploaded.getvalue())),
             caption=uploaded.name, use_container_width=True)

    # rune_champion 모듈 로드
    try:
        rc = importlib.import_module("rune_champion")
        with st.spinner("챔피언 10명 인식 중…"):
            champs10 = rc.extract_champions(str(temp_path))
        if champs10:
            st.success("인식된 챔피언: " + ", ".join(champs10))
        else:
            st.warning("챔피언을 인식하지 못했습니다. ROI/해상도를 확인하세요.")
    except Exception as e:
        st.exception(e)

# 2) 드롭다운으로 내 챔피언 선택
my_champion = st.selectbox("내 챔피언을 선택하세요 (인식된 10명 중에서)",
                           champs10, index=0) if champs10 else None
st.markdown("---")

# ROI 디버그 토글
if uploaded and champs10:
    try:
        rc = rc if "rc" in locals() else importlib.import_module("rune_champion")
        if st.checkbox("디버그: ROI 박스 표시"):
            img = rc.draw_rois(str(temp_path))
            st.image(img, caption="스케일된 ROI", use_container_width=True)
    except Exception as e:
        st.info(f"ROI 디버그 실패: {e}")

# ──────────────────────────────────────────────
# 3) 분석 시작
# ──────────────────────────────────────────────
go = st.button("분석 시작", disabled=not (temp_path and my_champion))

# ──────────────────────────────────────────────
# 4) 팀/적 정보 + 추천 이유 + 아이템 추천 표시
# ──────────────────────────────────────────────
if go:  # 분석 시작 버튼 클릭 시
    with st.spinner("분석 중…"):
        try:
            rc = rc if "rc" in locals() else importlib.import_module("rune_champion")
            my_team, enemy_team = rc.extract_champions_and_runes(str(temp_path), my_champion)

            st.subheader("팀/적 정보 확인")
            st.text(f"내 팀: {', '.join([c for c, _, _ in my_team])}")
            st.text(f"적 팀: {', '.join([c for c, _, _ in enemy_team])}")

            # 적 조합 요약 + 추천 이유
            num_ad, num_ap, num_tanks, num_support, num_cc = summarize_enemy(enemy_team)
            st.subheader("추천 이유:")
            for line in render_reasons(num_ad, num_ap, num_tanks, num_support, num_cc):
                st.write(f"- {line}")

            st.markdown("---")

            # === 아이템 빌드 추천 ===
            st.subheader("권장 아이템 빌드")
            recs = get_all_build_recommendations(my_champion, enemy_team)

            if not recs:
                st.info("해당 조합/역할에 대한 빌드가 없어요. (빌드 JSON 확인 필요)")
            else:
                for r in recs:
                    st.write(f"**{r['role']} 역할 추천 빌드**: {' → '.join(r['build'])}")

        except Exception as e:
            st.exception(e)
