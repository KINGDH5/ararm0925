# image.py — GitHub/Streamlit 배포용 (절대경로 제거, Secrets/ADC, 엔드포인트 캐싱)
from __future__ import annotations

import os
import io
import time
import base64
from typing import Tuple, Optional, List

import numpy as np
from PIL import Image, ImageDraw

# google-cloud
from google.cloud import aiplatform
from google.oauth2 import service_account

# ─────────────────────────────────────────────────────────────────────
# 좌표/스케일 설정
# ─────────────────────────────────────────────────────────────────────
BASE_W, BASE_H = 1920, 1080

# 블루팀 박스 (좌,상,우,하)
BLUE: List[Tuple[int, int, int, int]] = [
    (83, 157, 175, 248), (83, 277, 175, 368), (83, 397, 175, 488),
    (83, 517, 175, 608), (83, 637, 175, 728),
]

# 레드팀 박스 (좌,상,우,하)
RED: List[Tuple[int, int, int, int]] = [
    (529,16,602,89),(617,16,690,89),(705,16,778,89),(793,16,866,89),(881,16,954,89),
    (969,16,1042,89),(1057,16,1130,89),(1145,16,1218,89),(1233,16,1306,89),(1321,16,1394,89)
]


# ─────────────────────────────────────────────────────────────────────
# 좌표 유틸
# ─────────────────────────────────────────────────────────────────────
def _scale_coords(w: int, h: int, base: List[Tuple[int,int,int,int]],
                  dx: int = 0, dy: int = 0, sx: float = 1.0, sy: float = 1.0):
    rx, ry = (w / BASE_W) * sx, (h / BASE_H) * sy
    return [(int(l * rx) + dx, int(t * ry) + dy, int(r * rx) + dx, int(b * ry) + dy) for (l, t, r, b) in base]


def _crop(image: Image.Image, dx: int = 0, dy: int = 0, sx: float = 1.0, sy: float = 1.0):
    w, h = image.size
    b = _scale_coords(w, h, BLUE, dx, dy, sx, sy)
    r = _scale_coords(w, h, RED,  dx, dy, sx, sy)
    tiles = []
    for (l, t, rr, bb) in b + r:
        im = image.crop((l, t, rr, bb)).convert("RGB").resize((128, 128), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=50)
        tiles.append(buf.getvalue())
    return tiles, b, r


def draw_overlay(img: Image.Image, b: List[Tuple[int,int,int,int]], r: List[Tuple[int,int,int,int]]):
    im = img.copy()
    dr = ImageDraw.Draw(im)
    for x in b:
        dr.rectangle(x, outline=(0, 128, 255), width=3)
    for x in r:
        dr.rectangle(x, outline=(255, 64, 64), width=3)
    return im


# ─────────────────────────────────────────────────────────────────────
# Vertex 초기화 / 엔드포인트 캐시
# ─────────────────────────────────────────────────────────────────────
_ENDPOINT_CACHE = {}

def _creds_from_env_or_b64(b64_str: Optional[str] = None):
    """
    우선순위:
    1) 환경변수 GOOGLE_APPLICATION_CREDENTIALS 가 가리키는 JSON 파일
    2) Base64 인코딩된 서비스계정 JSON(문자열) — Streamlit Secrets로 전달된 경우
    3) None (ADC에 맡김; Cloud 런타임 서비스 계정 등)
    """
    # 1) 파일 경로(환경변수)
    cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if cred_path and os.path.exists(cred_path):
        try:
            return service_account.Credentials.from_service_account_file(cred_path)
        except Exception:
            pass  # fallthrough

    # 2) Base64 문자열
    if b64_str:
        try:
            raw = base64.b64decode(b64_str)
            return service_account.Credentials.from_service_account_info(__import__("json").loads(raw))
        except Exception:
            pass  # fallthrough

    # 3) None → ADC
    return None


def init_vertex(project_id: str, region: str, endpoint_id: str, credentials_b64: Optional[str] = None):
    """
    NOTE:
      - 기존 코드의 credentials_path 인자는 제거.
      - Streamlit Secrets의 GOOGLE_APPLICATION_CREDENTIALS_B64를 전달하거나,
        환경변수 GOOGLE_APPLICATION_CREDENTIALS 또는 ADC를 사용.
    """
    region_l = region.strip().lower()
    creds = _creds_from_env_or_b64(credentials_b64)

    key = (project_id, region_l, endpoint_id, bool(creds))
    if key in _ENDPOINT_CACHE:
        return _ENDPOINT_CACHE[key]

    aiplatform.init(
        project=project_id,
        location=region_l,
        credentials=creds,  # None이면 ADC 사용
        api_endpoint=f"{region_l}-aiplatform.googleapis.com",
    )
    ep = aiplatform.Endpoint(endpoint_id)
    _ENDPOINT_CACHE[key] = ep
    return ep


# ─────────────────────────────────────────────────────────────────────
# 예측
# ─────────────────────────────────────────────────────────────────────
def _predict_one(endpoint, img_bytes: bytes, retries: int = 3, delay: float = 0.5) -> Tuple[Optional[str], float]:
    inst = {"content": base64.b64encode(img_bytes).decode("utf-8")}
    last_err = None
    for _ in range(retries):
        try:
            resp = endpoint.predict(instances=[inst])
            # Vertex 예측 응답 포맷: {'predictions': [{'displayNames': [...], 'confidences': [...]}], ...}
            preds = getattr(resp, "predictions", None)
            if not preds:
                return (None, 0.0)
            pred = preds[0] or {}
            names = pred.get("displayNames", []) or pred.get("labels", [])
            confs = pred.get("confidences", []) or pred.get("scores", [])
            if not names or not confs:
                return (None, 0.0)
            i = int(np.argmax(confs))
            return (str(names[i]), float(confs[i]) * 100.0)
        except Exception as e:
            last_err = e
            time.sleep(delay)
    # 마지막 에러를 그대로 던져 상위에서 핸들하도록
    raise last_err


def predict_image(endpoint,
                  image: Image.Image,
                  threshold: float = 70.0,
                  dx: int = 0, dy: int = 0,
                  scale_w: float = 1.0, scale_h: float = 1.0):
    """
    반환값:
      current: 상위 5개(픽) 라벨 목록 (threshold 미만은 제외)
      bench:   뒤 10개(대기석) 라벨 목록 (Hwei/흐웨이 등은 None 처리)
      overlay: 박스가 그려진 PIL.Image
    """
    tiles, b, r = _crop(image, dx, dy, scale_w, scale_h)

    named = []
    for t in tiles:
        n, c = _predict_one(endpoint, t)
        named.append((n if (n and c >= threshold) else None, c if c >= threshold else 0.0))

    # 앞 5개 = 현재 픽
    current = [n for (n, _) in named[:5] if n]

    # 뒤 10개 = 대기석 (Hwei/null 처리)
    bench = []
    for (n, _) in named[5:]:
        if not n:
            bench.append(None)
        elif str(n).strip().lower() in {"hwei", "흐웨이"}:
            bench.append(None)
        else:
            bench.append(n)

    overlay = draw_overlay(image, b, r)
    return current, bench, overlay
