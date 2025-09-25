# ml.py — GitHub/Streamlit 배포용
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import numpy as np
import random
import warnings
import io

warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def read_csv_safe(path_or_buf):
    """
    CSV를 안전하게 읽기. 여러 인코딩 시도.
    path_or_buf: 파일 경로(str/Path) 또는 업로드된 파일 객체(BytesIO)
    """
    if isinstance(path_or_buf, (io.BytesIO, io.BufferedReader)):
        # 업로드된 파일 객체
        return pd.read_csv(path_or_buf, low_memory=False)
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr", "latin1"]:
        try:
            return pd.read_csv(path_or_buf, encoding=enc, low_memory=False)
        except Exception:
            continue
    return pd.read_csv(path_or_buf, low_memory=False)


def train_models(df, verbose: bool = True):
    champ_cols = [f'champ{i}_name' for i in range(1, 6)]

    # --- Synergy ---
    all_champs = sorted(pd.unique(df[champ_cols].values.ravel("K")).tolist())
    mlb = MultiLabelBinarizer(classes=all_champs)
    X_onehot = pd.DataFrame(mlb.fit_transform(df[champ_cols].values.tolist()), columns=mlb.classes_)
    y = df["win"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_onehot, y, test_size=0.2, random_state=SEED, stratify=y
    )
    synergy_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=SEED,
    )
    synergy_model.fit(X_train, y_train)
    synergy_acc = accuracy_score(y_test, synergy_model.predict(X_test))

    # --- Champion-wise ---
    AURA_KEYS = [
        "damage_dealt", "damage_taken", "attack_speed", "skill_haste",
        "hp_regen", "tenacity", "shield_absorb", "energy_regen",
    ]
    long_rows = []
    for _, row in df.iterrows():
        for i in range(1, 6):
            champ = row[f"champ{i}_name"]
            if pd.isna(champ):
                continue
            item = {"champion": champ, "win": int(row["win"])}
            for key in AURA_KEYS:
                col_name = f"champ{i}_name_{key}"
                item[key] = row.get(col_name, 0.0)
            item["CCcount"] = row.get(f"champ{i}_name_CCcount", 0.0)
            for role in ["ad_items", "ap_items", "tank_items", "ranged"]:
                col = f"champ{i}_is_{role}"
                item[col] = row.get(col, 0)
            long_rows.append(item)

    champ_long = pd.DataFrame(long_rows).fillna(0.0)
    champ_feature_cols = [
        c
        for c in (
            AURA_KEYS
            + ["CCcount", "champ1_is_ad_items", "champ1_is_ap_items",
               "champ1_is_tank_items", "champ1_is_ranged"]
        )
        if c in champ_long.columns
    ]
    Xc = champ_long[champ_feature_cols]
    yc = champ_long["win"]
    champ_model = xgb.XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1, random_state=SEED, eval_metric="logloss"
    )
    champ_model.fit(Xc, yc)
    champ_profile = champ_long.groupby("champion")[champ_feature_cols].median().reset_index()
    champ_acc = accuracy_score(yc, champ_model.predict(Xc))

    # --- Stat/Tag ---
    stat_types = ["hp", "mp", "armor", "spellblock", "attackdamage", "attackspeed"]
    lvl_suffixes = ["_lvl3", "_lvl6", "_lvl11", "_lvl16", "_lvl18"]
    lvl_cols = [
        f"{s}{lvl}"
        for s in stat_types
        for lvl in lvl_suffixes
        if f"{s}{lvl}" in df.columns
    ]

    tag_cols = [f"champ{i}_tags" for i in range(1, 6)]
    df["all_tags"] = df[tag_cols].fillna("").astype(str).agg(",".join, axis=1)

    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(","))
    tag_matrix = vectorizer.fit_transform(df["all_tags"])
    tag_df = pd.DataFrame(tag_matrix.toarray(), columns=[f"tag_{t}" for t in vectorizer.get_feature_names_out()])

    feature_cols = lvl_cols + list(tag_df.columns)
    X_stat = pd.concat([df[lvl_cols], tag_df], axis=1)
    y_stat = df["win"]
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_stat, y_stat, test_size=0.2, random_state=SEED, stratify=y_stat
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_s)

    stat_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, random_state=SEED, eval_metric="logloss"
    )
    stat_model.fit(X_train_scaled, y_train_s)
    stat_acc = accuracy_score(y_test_s, stat_model.predict(X_test_s))

    if verbose:
        print(f"[Synergy 모델 정확도]: {synergy_acc:.2%}")
        print(f"[챔피언 개별 모델 정확도]: {champ_acc:.2%}")
        print(f"[스탯/태그 모델 정확도]: {stat_acc:.2%}")

    return synergy_model, champ_model, mlb, champ_profile, stat_model, scaler, feature_cols, vectorizer, df, champ_cols


def get_team_winrate(team_champs, models):
    synergy_model, champ_model, mlb, champ_profile, stat_model, scaler, feature_cols, vectorizer, df, champ_cols = models

    # Synergy
    onehot_vec = mlb.transform([team_champs])
    p_synergy = synergy_model.predict_proba(onehot_vec)[0][1]

    # Champ-wise
    scores = []
    for cand in team_champs:
        r = champ_profile[champ_profile["champion"] == cand]
        if not r.empty:
            p = champ_model.predict_proba(r[[c for c in champ_profile.columns if c != "champion"]])[0][1]
        else:
            p = 0.5
        scores.append(p)
    p_champ = sum(scores) / len(scores)

    # Stat/Tag
    team_rows = df[df[champ_cols].apply(lambda row: any(c in row.values for c in team_champs), axis=1)]
    lvl_cols = [c for c in feature_cols if not c.startswith("tag_")]
    team_avg_stats = team_rows[lvl_cols].mean()
    team_tag_texts = team_rows[champ_cols].fillna("").astype(str).agg(",".join, axis=1)
    tag_matrix_candidate = vectorizer.transform(team_tag_texts)
    tag_df_candidate = pd.DataFrame(tag_matrix_candidate.toarray(), columns=[f"tag_{t}" for t in vectorizer.get_feature_names_out()])
    tag_mean = tag_df_candidate.mean()
    feature_vector = pd.concat([team_avg_stats, tag_mean])
    fv_dict = {col: float(feature_vector.get(col, 0.0)) for col in feature_cols}
    fv_df = pd.DataFrame([fv_dict])
    fv_scaled = scaler.transform(fv_df)
    p_stat = stat_model.predict_proba(fv_scaled)[0][1]

    # 가중합
    return 0.6 * p_synergy + 0.25 * p_stat + 0.15 * p_champ


def list_all_champs(models):
    """UI용 편의 함수"""
    _, _, mlb, _, _, _, _, _, _, _ = models
    return list(mlb.classes_)
