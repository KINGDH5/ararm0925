# app.py — GitHub/Streamlit 배포용 (절대경로 제거, Secrets 지원, 엔드포인트 캐싱)

import os
import base64
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
from PIL import Image

from ml import read_csv_safe, train_models, get_team_winrate, list_all_champs
from image import init_vertex, predict_image


# ----------------------------
# 경로/설정 (절대경로 제거)
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "data" / "renamed_data.csv"   # ← 레포 내 data/ 폴더 사용

# ---- Vertex 설정: Streamlit Secrets → 없으면 기능 비활성 ----
PROJECT_ID  = st.secrets.get("PROJECT_ID")
REGION      = st.secrets.get("REGION", "us-central1")
ENDPOINT_ID = st.secrets.get("ENDPOINT_ID")
CREDS_B64   = st.secrets.get("GOOGLE_APPLICATION_CREDENTIALS_B64")  # 선택


def _ensure_adc_from_b64():
    """secrets에 서비스계정 JSON(Base64)이 있으면 임시파일로 복원해 ADC 설정."""
    if not CREDS_B64:
        return None
    try:
        raw = base64.b64decode(CREDS_B64)
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        tf.write(raw); tf.flush(); tf.close()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tf.name
        return tf.name
    except Exception as e:
        st.warning(f"크리덴셜 복원 실패: {e}")
        return None


# === EN->KO 챔피언 매핑 ===
EN2KO = {
    "Aatrox":"아트록스","Ahri":"아리","Akali":"아칼리","Akshan":"아크샨","Alistar":"알리스타",
    "Amumu":"아무무","Anivia":"애니비아","Annie":"애니","Aphelios":"아펠리오스","Ashe":"애쉬",
    "Aurelion Sol":"아우렐리온 솔","Aurora":"오로라","Azir":"아지르",
    "Bard":"바드","Bel'Veth":"벨베스","Blitzcrank":"블리츠크랭크","Brand":"브랜드","Braum":"브라움",
    "Caitlyn":"케이틀린","Camille":"카밀","Cassiopeia":"카시오페아","Cho'Gath":"초가스","Corki":"코르키",
    "Darius":"다리우스","Diana":"다이애나","Dr. Mundo":"문도 박사","Draven":"드레이븐",
    "Ekko":"에코","Elise":"엘리스","Evelynn":"이블린","Ezreal":"이즈리얼",
    "Fiddlesticks":"피들스틱","Fiora":"피오라","Fizz":"피즈",
    "Galio":"갈리오","Gangplank":"갱플랭크","Garen":"가렌","Gnar":"나르","Gragas":"그라가스","Graves":"그레이브즈","Gwen":"그웬",
    "Hecarim":"헤카림","Heimerdinger":"하이머딩거",
    "Illaoi":"일라오이","Irelia":"이렐리아","Ivern":"아이번",
    "Janna":"잔나","Jarvan IV":"자르반 4세","Jax":"잭스","Jayce":"제이스","Jhin":"진","Jinx":"징크스",
    "K'Sante":"크산테","Kai'Sa":"카이사","Kalista":"칼리스타","Karma":"카르마","Karthus":"카서스","Kassadin":"카사딘",
    "Katarina":"카타리나","Kayle":"케일","Kayn":"케인","Kennen":"케넨","Kha'Zix":"카직스","Kindred":"킨드레드",
    "Kled":"클레드","Kog'Maw":"코그모",
    "LeBlanc":"르블랑","Lee Sin":"리 신","Leona":"레오나","Lillia":"릴리아","Lissandra":"리산드라",
    "Lucian":"루시안","Lulu":"룰루","Lux":"럭스",
    "Malphite":"말파이트","Malzahar":"말자하","Maokai":"마오카이","Master Yi":"마스터 이",
    "Milio":"밀리오","Miss Fortune":"미스 포츈","Mordekaiser":"모데카이저","Morgana":"모르가나",
    "Naafiri":"나피리","Nami":"나미","Nasus":"나서스","Nautilus":"노틸러스","Neeko":"니코","Nidalee":"니달리","Nilah":"닐라",
    "Nocturne":"녹턴","Nunu & Willump":"누누와 윌럼프",
    "Olaf":"올라프","Orianna":"오리아나","Ornn":"오른",
    "Pantheon":"판테온","Poppy":"뽀삐","Pyke":"파이크",
    "Qiyana":"키아나","Quinn":"퀸",
    "Rakan":"라칸","Rammus":"람머스","Rek'Sai":"렉사이","Rell":"렐","Renata Glasc":"레나타 글라스크",
    "Renekton":"레넥톤","Rengar":"렝가","Riven":"리븐","Rumble":"럼블","Ryze":"라이즈",
    "Samira":"사미라","Sejuani":"세주아니","Senna":"세나","Seraphine":"세라핀","Sett":"세트","Shaco":"샤코","Shen":"쉔",
    "Shyvana":"쉬바나","Singed":"신지드","Sion":"사이온","Sivir":"시비르","Skarner":"스카너","Smolder":"스몰더","Sona":"소나",
    "Soraka":"소라카","Swain":"스웨인","Sylas":"사일러스","Syndra":"신드라",
    "Tahm Kench":"탐 켄치","Taliyah":"탈리야","Talon":"탈론","Taric":"타릭","Teemo":"티모","Thresh":"쓰레쉬",
    "Tristana":"트리스타나","Trundle":"트런들","Tryndamere":"트린다미어","Twisted Fate":"트위스티드 페이트","Twitch":"트위치",
    "Udyr":"우디르","Urgot":"우르곳",
    "Varus":"바루스","Vayne":"베인","Veigar":"베이가","Vel'Koz":"벨코즈","Vex":"벡스","Vi":"바이","Viego":"비에고",
    "Viktor":"빅토르","Vladimir":"블라디미르","Volibear":"볼리베어",
    "Warwick":"워윅","Wukong":"오공","Xayah":"자야","Xerath":"제라스","Xin Zhao":"신 짜오",
    "Yasuo":"야스오","Yone":"요네","Yorick":"요릭","Yuumi":"유미",
    "Zac":"자크","Zed":"제드","Zeri":"제리","Ziggs":"직스","Zilean":"질리언","Zoe":"조이","Zyra":"자이라",
}

def _map_and_filter_detected(names_list, options):
    opt = set(options or [])
    out = []
    for n in names_list or []:
        k = EN2KO.get(n, n)
        if k in opt and k not in out:
            out.append(k)
    return out


st.set_page_config(page_title="ARAM 픽 최적화", layout="wide")
st.title("⭐ ARAM 픽 최적화 (같은 모델 값, 파일만 3개)")

# ----------------------------
# 1) CSV 선택
# ----------------------------
st.sidebar.header("데이터")
mode = st.sidebar.radio("CSV", ["기본 경로", "파일 업로드"], horizontal=True)
df = None

if mode == "기본 경로":
    st.sidebar.caption(f"기본 경로: {DEFAULT_CSV}")
    if DEFAULT_CSV.exists():
        df = read_csv_safe(str(DEFAULT_CSV))
    else:
        st.sidebar.warning("data/renamed_data.csv 가 레포에 없습니다.")
else:
    up = st.sidebar.file_uploader("CSV 업로드", type=["csv"])
    if up:
        df = read_csv_safe(up)

if df is None:
    st.info("CSV를 선택하세요.")
    st.stop()

st.dataframe(df.head(3), use_container_width=True)

# ----------------------------
# 2) 학습
# ----------------------------
if "models" not in st.session_state:
    st.session_state.models = None

if st.button("학습 시작 / 다시 학습", type="primary"):
    with st.spinner("학습 중..."):
        st.session_state.models = train_models(df)

if not st.session_state.models:
    st.stop()

models = st.session_state.models
all_champs = list_all_champs(models)

# ----------------------------
# 3) (선택) 스크린샷 감지
# ----------------------------
st.sidebar.subheader("스크린샷 감지 (옵션)")
use_vertex = st.sidebar.checkbox("사용", value=False)
threshold = st.sidebar.slider("신뢰도(%)", 50, 95, 50, 1)

@st.cache_resource
def get_endpoint():
    """Vertex 엔드포인트 초기화(옵션). secrets 없으면 None."""
    if not (PROJECT_ID and ENDPOINT_ID):
        return None
    _ensure_adc_from_b64()
    try:
        # 새 시그니처(3개 인자)
        return init_vertex(PROJECT_ID, REGION, ENDPOINT_ID)
    except TypeError:
        # 예전 시그니처(4개 인자) 호환
        return init_vertex(PROJECT_ID, REGION, ENDPOINT_ID, None)

detected_current, detected_bench = [], []
uploaded = st.file_uploader("픽 화면 스크린샷 (png/jpg)", type=["png","jpg","jpeg"]) if use_vertex else None

if uploaded and use_vertex:
    endpoint = get_endpoint()
    if endpoint is None:
        st.warning("Streamlit Secrets에 PROJECT_ID/ENDPOINT_ID가 없어 감지 기능이 비활성화되었습니다.")
    else:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="업로드 이미지", use_container_width=True)
        with st.spinner("감지 중..."):
            cur, bench, overlay = predict_image(endpoint, image, threshold=threshold)
        st.image(overlay, caption="탐지 영역", use_container_width=True)
        detected_current = _map_and_filter_detected(cur, all_champs)[:5]
        detected_bench = _map_and_filter_detected(bench, all_champs)[:10]

# ----------------------------
# 4) 우리 팀 5명 선택
# ----------------------------
default_team = (detected_current if len(detected_current) == 5 else all_champs[:5])
my_team = st.multiselect("우리 팀 (5명)", options=all_champs, default=default_team, max_selections=5)

if len(my_team) != 5:
    st.warning("5명을 선택하세요.")
    st.stop()

wr = get_team_winrate(my_team, models)
st.markdown(f"### 현재 픽 승률: **{wr*100:.2f}%**")

# ----------------------------
# 5) 교체 추천
# ----------------------------
pool = st.multiselect(
    "교체 후보",
    options=[c for c in all_champs if c not in my_team],
    default=[c for c in detected_bench if c not in my_team],
)

target = st.selectbox("교체할 내 챔피언", options=my_team)
rows = []
best = None
best_inc = 0.0

for cand in pool:
    new_team = [cand if x == target else x for x in my_team]
    w = get_team_winrate(new_team, models)
    inc = w - wr
    rows.append({"교체 챔피언": cand, "새 승률(%)": round(w * 100, 2), "변화량 Δ(%)": round(inc * 100, 2)})
    if inc > best_inc:
        best, best_inc = (target, cand, w), inc

if rows:
    st.dataframe(pd.DataFrame(rows).sort_values("새 승률(%)", ascending=False), use_container_width=True)
if best:
    st.success(f"🔷 {best[0]} → {best[1]} 교체 시 **{best[2]*100:.2f}%**")
else:
    st.info("교체 후보를 선택하면 추천이 표시됩니다.")
