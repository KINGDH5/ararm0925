# rune_champion.py — Streamlit Cloud 안전 실행판
# -*- coding: utf-8 -*-
import os, io, base64, json
from pathlib import Path
from PIL import Image, ImageEnhance, ImageDraw
import pandas as pd

# ─────────────────────────────────────────────
# Streamlit secrets 지원
# ─────────────────────────────────────────────
try:
    import streamlit as st
    _SECRETS = dict(st.secrets) if hasattr(st, "secrets") else {}
except Exception:
    _SECRETS = {}

def _get_secret(key, default=None):
    return os.environ.get(key) or _SECRETS.get(key, default)

# ─────────────────────────────────────────────
# Google SDK 임포트 (없어도 앱 죽지 않게)
# ─────────────────────────────────────────────
try:
    from google.cloud import vision, aiplatform
    from google.oauth2 import service_account
    _GOOGLE_OK = True
except Exception:
    _GOOGLE_OK = False
    vision = aiplatform = service_account = None

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

# ─────────────────────────────────────────────
# GCP 인증 로더
# ─────────────────────────────────────────────
def _load_creds_from_env_or_b64(b64_key: str = None, file_hint: str = None):
    if not _GOOGLE_OK:
        return None
    try:
        cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if cred_path and Path(cred_path).exists():
            return service_account.Credentials.from_service_account_file(cred_path)
        if file_hint and Path(file_hint).exists():
            return service_account.Credentials.from_service_account_file(file_hint)
        if b64_key:
            b64_val = _get_secret(b64_key)
            if b64_val:
                data = json.loads(base64.b64decode(b64_val))
                return service_account.Credentials.from_service_account_info(data)
    except Exception:
        pass
    return None

# ─────────────────────────────────────────────
# Vision / Vertex 안전 초기화
# ─────────────────────────────────────────────
VISION_CRED = _load_creds_from_env_or_b64("VISION_CRED_B64")
if _GOOGLE_OK:
    try:
        vision_client = vision.ImageAnnotatorClient(credentials=VISION_CRED)
    except Exception:
        vision_client = None
else:
    vision_client = None

RUNE_PROJECT_ID  = _get_secret("RUNE_PROJECT_ID",  "your-project-id")
RUNE_LOCATION    = _get_secret("RUNE_LOCATION",    "us-central1")
RUNE_ENDPOINT_ID = _get_secret("RUNE_ENDPOINT_ID", "0000000000000000000")
RUNE_CRED        = _load_creds_from_env_or_b64("RUNE_CRED_B64")

if _GOOGLE_OK:
    try:
        aiplatform.init(project=RUNE_PROJECT_ID, location=RUNE_LOCATION, credentials=RUNE_CRED)
        RUNE_endpoint = aiplatform.Endpoint(RUNE_ENDPOINT_ID)
    except Exception:
        RUNE_endpoint = None
else:
    RUNE_endpoint = None

# ─────────────────────────────────────────────
# 기준 해상도 / 스케일
# ─────────────────────────────────────────────
BASE_W, BASE_H = 1920, 1080
def _scale_box(box, img_w, img_h):
    x1, y1, x2, y2 = box
    sx, sy = img_w / BASE_W, img_h / BASE_H
    return (int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy))

# 챔피언/룬 박스 좌표
champion_name_regions = [
    (243, 407, 487, 433), (539, 407, 783, 433), (834, 407, 1080, 433),
    (1128, 407, 1375, 433), (1425, 407, 1671, 433), (243, 928, 487, 958),
    (539, 928, 783, 958), (834, 928, 1080, 958), (1128, 928, 1375, 958),
    (1425, 928, 1671, 958)
]
RUNE_boxes = [
    (255, 447, 281, 474), (551, 447, 577, 474), (847, 447, 873, 474),
    (1144, 447, 1169, 474), (1440, 447, 1466, 474), (255, 971, 281, 999),
    (551, 971, 577, 999), (847, 971, 873, 999), (1144, 971, 1169, 999),
    (1440, 971, 1466, 999)
]

# 룬 이름 매핑
RUNE_NAME_MAP = {
    "Electrocute": "감전", "Predator": "포식자", "DarkHarvest": "어둠의 수확",
    "HailOfBlades": "칼날비", "GlacialAugment": "빙결 강화",
    "UnsealedSpellbook": "봉인 풀린 주문서", "FirstStrike": "선제공격",
    "PressTheAttack": "집중 공격", "LethalTempo": "치명적 속도",
    "FleetFootwork": "기민한 발놀림", "Conqueror": "정복자",
    "GraspOfTheUndying": "착취의 손아귀", "Aftershock": "여진",
    "Guardian": "수호자", "SummonAery": "콩콩이 소환",
    "ArcaneComet": "신비로운 유성", "PhaseRush": "난입",
}

# CSV 로드
champion_df = pd.read_csv(BASE_DIR / "lol_champions.csv", encoding="utf-8")
champions_list = champion_df["name"].tolist()
role_df = pd.read_csv(BASE_DIR / "champion_rune_roles.csv", encoding="utf-8")

def get_role(champ_name, rune_name):
    row = role_df[role_df["name"] == champ_name]
    if not row.empty and rune_name in row.columns:
        return row.iloc[0][rune_name]
    return "정보 없음"

# ─────────────────────────────────────────────
# OCR 함수
# ─────────────────────────────────────────────
def ocr_champion_region(image_path, region):
    if vision_client is None:
        return ""
    with Image.open(image_path) as img:
        w, h = img.size
        r = _scale_box(region, w, h)
        cropped = ImageEnhance.Contrast(img.crop(r).convert("L")).enhance(2.0)
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
    image = vision.Image(content=buf.getvalue())
    resp = vision_client.text_detection(image=image)
    if resp.error.message:
        raise Exception(f"OCR Error: {resp.error.message}")
    texts = resp.text_annotations
    return texts[0].description.strip() if texts else ""

NAME_CORRECTION = {"오콩": "오공"}

def extract_champions(image_path):
    out = []
    for region in champion_name_regions:
        text = ocr_champion_region(image_path, region)
        text = NAME_CORRECTION.get(text, text)
        matched = [c for c in champions_list if c in text]
        out.append(matched[0] if matched else text)
    return out

# ─────────────────────────────────────────────
# 룬 예측 (Vertex)
# ─────────────────────────────────────────────
def predict_RUNE(endpoint, image_bytes, threshold=35.0):
    if endpoint is None:
        return "null", 0.0
    inst = {"content": base64.b64encode(image_bytes).decode("utf-8")}
    try:
        res = endpoint.predict(instances=[inst])
        preds = getattr(res, "predictions", None)
        if not preds:
            return "null", 0.0
        pred = preds[0]
        if isinstance(pred, dict):
            names = pred.get("displayNames", []); confs = pred.get("confidences", [])
            if names and confs:
                i = int(max(range(len(confs)), key=lambda j: confs[j]))
                best = RUNE_NAME_MAP.get(names[i], names[i])
                conf = confs[i] * 100
                return (best if conf >= threshold else "null", conf)
        return str(pred), 0.0
    except Exception as e:
        print(f"예측 오류: {e}")
        return "null", 0.0

def crop_and_predict_RUNEs(image_path):
    results = []
    with Image.open(image_path) as img:
        w, h = img.size
        for box in RUNE_boxes:
            b = _scale_box(box, w, h)
            cropped = img.crop(b).convert("RGB").resize((64,64), Image.Resampling.LANCZOS)
            buf = io.BytesIO(); cropped.save(buf, format="JPEG", quality=90)
            results.append(predict_RUNE(RUNE_endpoint, buf.getvalue()))
    return results

# ─────────────────────────────────────────────
# 팀/룬/역할군 추출
# ─────────────────────────────────────────────
def extract_champions_and_runes(image_path, my_champion):
    champions = extract_champions(image_path)
    runes = crop_and_predict_RUNEs(image_path)
    try:
        my_index = champions.index(my_champion)
    except ValueError:
        raise ValueError(f"인식된 10명에 '{my_champion}' 없음. 인식값: {champions}")
    if my_index < 5:
        my_team_champs, enemy_team_champs = champions[:5], champions[5:]
        my_team_runes,  enemy_team_runes  = runes[:5],      runes[5:]
    else:
        my_team_champs, enemy_team_champs = champions[5:], champions[:5]
        my_team_runes,  enemy_team_runes  = runes[5:],      runes[:5]
    my_team    = [(c, r[0], get_role(c, r[0])) for c, r in zip(my_team_champs,    my_team_runes)]
    enemy_team = [(c, r[0], get_role(c, r[0])) for c, r in zip(enemy_team_champs, enemy_team_runes)]
    return my_team, enemy_team

# ─────────────────────────────────────────────
# ROI 디버그
# ─────────────────────────────────────────────
def draw_rois(image_path, save_path=None):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size; dr = ImageDraw.Draw(img)
    for r in champion_name_regions:
        x1,y1,x2,y2 = _scale_box(r,w,h); dr.rectangle([x1,y1,x2,y2], outline=(255,0,0), width=3)
    for r in RUNE_boxes:
        x1,y1,x2,y2 = _scale_box(r,w,h); dr.rectangle([x1,y1,x2,y2], outline=(255,255,0), width=2)
    if save_path: img.save(save_path)
    return img
