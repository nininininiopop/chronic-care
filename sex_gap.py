"""기준 모델 vs 확장 모델 비교 (고혈압·당뇨병 의료이용률 2024)

실행 순서:
    python preprocess.py   # data/processed/ 생성
    python model.py        # output/ 에 표·그래프 저장

모델 (모두 반응변수 = 분자/분모 이항 모델)
    M0 기준          : 연령 + 성별
    M1 +교호작용     : M0 + 연령×성별
    M2 +시도         : M1 + 시도
    M3 +지역특성     : M1 + 행정유형(구/시/군) + 고령비중(70세 이상)
    M4 확장(전체)    : M1 + 시도 + 행정유형 + 고령비중
    RF 랜덤포레스트  : 연령 + 성별 + 시도 + 행정유형 + 고령비중 (비교용)

평가: 시군구 단위 GroupKFold 교차검증(같은 시군구가 학습·평가에 섞이지 않음)
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy.special import expit, xlogy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

BASE_DIR = Path(__file__).resolve().parent
if BASE_DIR.name in ("scripts", "pages"):  # scripts/ 또는 pages/ 에 두어도 프로젝트 루트 기준으로 동작
    BASE_DIR = BASE_DIR.parent
PROC_DIR = BASE_DIR / "data" / "processed"
OUT_DIR = BASE_DIR / "output"

SEED = 42
N_SPLITS = 5
SMALL_N = 1000  # 분모가 이 값 미만이면 소표본
AGE_ORDER = ["0~9세", "10대", "20대", "30대", "40대", "50대", "60대", "70대", "80세 이상"]


# ------------------------------------------------------------------ 준비
def setup_font() -> None:
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Malgun Gothic", "AppleGothic", "NanumGothic", "NanumBarunGothic",
                 "Noto Sans CJK KR", "Noto Sans KR"):
        if cand in names:
            plt.rcParams["font.family"] = cand
            plt.rcParams["axes.unicode_minus"] = False
            return
    print("⚠ 한글 폰트를 찾지 못했어요. 그래프의 한글이 깨질 수 있습니다. (나눔고딕 설치 권장)")


def load() -> pd.DataFrame:
    mp, fp = PROC_DIR / "model_input.csv", PROC_DIR / "region_features.csv"
    if not mp.exists() or not fp.exists():
        sys.exit("data/processed/ 파일이 없습니다. 먼저 `python preprocess.py`를 실행하세요.")
    df = pd.read_csv(mp)
    feat = pd.read_csv(fp)[["지역", "고령비중_70세이상"]]
    df = df.merge(feat, on="지역", how="left", validate="many_to_one")
    # 시군구 이름 끝 글자로 행정유형 구분 (군=농어촌, 구=도시 자치구·일반구, 시=그 외)
    df["행정유형"] = df["시군구"].str[-1].where(df["시군구"].str[-1].isin(["구", "시", "군"]), "기타")
    return df.reset_index(drop=True)


def build_blocks(df: pd.DataFrame) -> dict:
    """설명변수 묶음을 한 번만 만들어 모델별로 조합한다."""
    age_cat = df["연령"].astype(pd.CategoricalDtype(AGE_ORDER))
    age = pd.get_dummies(age_cat, prefix="연령", drop_first=True, dtype=float)
    sex = df["성별"].eq("여성").astype(float).rename("여성")
    inter = age.mul(sex, axis=0).add_prefix("여성×")
    sido = pd.get_dummies(df["시도"], prefix="시도", drop_first=True, dtype=float)
    kind = pd.get_dummies(df["행정유형"], prefix="유형", drop_first=True, dtype=float)
    aging = ((df["고령비중_70세이상"] - df["고령비중_70세이상"].mean())
             / df["고령비중_70세이상"].std()).rename("고령비중_z")
    return {
        "const": pd.DataFrame({"const": 1.0}, index=df.index),
        "age": age, "sex": sex.to_frame(), "inter": inter,
        "sido": sido, "kind": kind, "aging": aging.to_frame(),
        "age_num": df[["연령_수치"]].astype(float),
    }


# ------------------------------------------------------------------ 이항 GLM (IRLS)
def fit_binom(X: np.ndarray, y: np.ndarray, n: np.ndarray, ridge: float = 1e-6):
    """로짓 연결 이항 GLM. 반환: 계수, (X'WX)^-1"""
    b = np.zeros(X.shape[1])
    R = np.eye(X.shape[1]) * ridge
    R[0, 0] = 0.0  # 절편은 규제하지 않음
    for _ in range(100):
        eta = X @ b
        p = np.clip(expit(eta), 1e-10, 1 - 1e-10)
        w = n * p * (1 - p)
        z = eta + (y - n * p) / w
        xtw = X.T * w
        b_new = np.linalg.solve(xtw @ X + R, xtw @ z)
        done = np.max(np.abs(b_new - b)) < 1e-9
        b = b_new
        if done:
            break
    p = np.clip(expit(X @ b), 1e-10, 1 - 1e-10)
    w = n * p * (1 - p)
    cov = np.linalg.inv((X.T * w) @ X + R)
    return b, cov


def pearson_dispersion(y, n, p, n_params) -> float:
    p = np.clip(p, 1e-10, 1 - 1e-10)
    chi2 = (((y - n * p) ** 2) / (n * p * (1 - p))).sum()
    return chi2 / (len(y) - n_params)


# ------------------------------------------------------------------ 평가
def deviance_rows(y, n, p) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return 2 * (xlogy(y, y / (n * p)) + xlogy(n - y, (n - y) / (n * (1 - p))))


def region_ratio(df: pd.DataFrame, p: np.ndarray) -> pd.Series:
    """지역별 실제 이용자 ÷ 기대 이용자. 1보다 크면 모델 예측보다 이용이 많은 지역."""
    g = df.assign(E=df["분모"].to_numpy() * p).groupby("지역").agg(O=("분자", "sum"), E=("E", "sum"))
    return g["O"] / g["E"]


def evaluate(df: pd.DataFrame, p: np.ndarray) -> dict:
    y, n = df["분자"].to_numpy(float), df["분모"].to_numpy(float)
    err = (p - y / n) * 100  # %p
    small = n < SMALL_N
    return {
        "MAE(%p)": np.abs(err).mean(),
        "가중MAE(%p)": np.average(np.abs(err), weights=n),
        "RMSE(%p)": np.sqrt((err ** 2).mean()),
        "소표본MAE(%p)": np.abs(err[small]).mean(),
        "평균편차": deviance_rows(y, n, p).mean(),
        "지역잔차SD": region_ratio(df, p).std(ddof=1),
    }


def cv_predict(fit_predict, n_rows: int, groups: np.ndarray) -> np.ndarray:
    oof = np.full(n_rows, np.nan)
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(np.zeros(n_rows), groups=groups):
        oof[te] = fit_predict(tr, te)
    assert not np.isnan(oof).any()
    return oof


# ------------------------------------------------------------------ 본 실행
def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    setup_font()
    df = load()
    y, n = df["분자"].to_numpy(float), df["분모"].to_numpy(float)
    groups = df["지역"].to_numpy()
    blocks = build_blocks(df)
    print(f"데이터: {len(df):,}행 · 시군구 {df['지역'].nunique()}곳 · {N_SPLITS}-fold 그룹 교차검증\n")

    glm_specs = {
        "M0 기준(연령+성별)": ["age", "sex"],
        "M1 +연령×성별": ["age", "sex", "inter"],
        "M2 +시도": ["age", "sex", "inter", "sido"],
        "M3 +행정유형·고령비중": ["age", "sex", "inter", "kind", "aging"],
        "M4 확장(전체)": ["age", "sex", "inter", "sido", "kind", "aging"],
    }

    rows, oof_all, dispersions, n_params = [], {}, {}, {}
    full_p0 = None
    last_fit = None

    for name, keys in glm_specs.items():
        X = pd.concat([blocks["const"]] + [blocks[k] for k in keys], axis=1)
        Xv = X.to_numpy(float)

        def fit_predict(tr, te, Xv=Xv):
            b, _ = fit_binom(Xv[tr], y[tr], n[tr])
            return expit(Xv[te] @ b)

        oof = cv_predict(fit_predict, len(df), groups)
        b, cov = fit_binom(Xv, y, n)
        p_full = expit(Xv @ b)
        oof_all[name] = oof
        n_params[name] = Xv.shape[1]
        dispersions[name] = pearson_dispersion(y, n, p_full, Xv.shape[1])
        rows.append({"모델": name, "파라미터수": Xv.shape[1], **evaluate(df, oof)})
        if name.startswith("M0"):
            full_p0 = p_full
        last_fit = (X.columns, b, cov, dispersions[name])

    # 랜덤포레스트 (비교용)
    X_rf = pd.concat([blocks[k] for k in ["age_num", "sex", "sido", "kind", "aging"]], axis=1).to_numpy(float)

    def rf_predict(tr, te):
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1)
        rf.fit(X_rf[tr], (y / n)[tr], sample_weight=n[tr])
        return np.clip(rf.predict(X_rf[te]), 1e-6, 1 - 1e-6)

    oof_all["RF 랜덤포레스트"] = cv_predict(rf_predict, len(df), groups)
    rows.append({"모델": "RF 랜덤포레스트", "파라미터수": np.nan, **evaluate(df, oof_all["RF 랜덤포레스트"])})

    res = pd.DataFrame(rows)
    base_dev = res.loc[0, "평균편차"]
    res["편차감소율(%)"] = (base_dev - res["평균편차"]) / base_dev * 100
    res["과산포(전체적합)"] = [dispersions.get(m, np.nan) for m in res["모델"]]
    res.to_csv(OUT_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 200)
    print("=== 교차검증(시군구 단위) 성능 비교 ===")
    print(res.round(3).to_string(index=False))

    # ---- 질문 1: 지역 격차 중 연령·성별 구성으로 설명되는 부분 (기술통계, 전체 적합)
    reg = df.groupby("지역")[["분자", "분모"]].sum()
    crude = (reg["분자"] / reg["분모"]) / (reg["분자"].sum() / reg["분모"].sum())
    adj = region_ratio(df, full_p0)
    share = 1 - adj.var(ddof=1) / crude.var(ddof=1)
    print("\n=== 지역 격차 분해 (시군구 252곳, 전국 대비 비율의 표준편차) ===")
    print(f"보정 전(단순 비율)      SD = {crude.std(ddof=1):.3f}")
    print(f"연령·성별 보정 후(M0)   SD = {adj.std(ddof=1):.3f}")
    print(f"→ 지역 간 변동의 약 {share * 100:.0f}%가 연령·성별 구성 차이로 설명됨 (분산 기준)")

    # ---- 질문 2: 보정 후 남는 지역 격차 (M0 기준 실제/기대)
    best = res.loc[res["평균편차"].idxmin(), "모델"]
    gap = pd.DataFrame({
        "시도": df.drop_duplicates("지역").set_index("지역")["시도"],
        "실제이용자": reg["분자"],
        "단순비율": crude,
        "보정후비율_M0": adj,
        f"보정후비율_{best.split()[0]}(교차검증)": region_ratio(df, oof_all[best]),
    }).sort_values("보정후비율_M0")
    gap.to_csv(OUT_DIR / "region_gap.csv", encoding="utf-8-sig")
    print("\n=== 연령·성별 보정 후에도 이용이 적은/많은 지역 (실제÷기대, M0) ===")
    print("낮음:", ", ".join(f"{i}({v:.2f})" for i, v in gap["보정후비율_M0"].head(5).items()))
    print("높음:", ", ".join(f"{i}({v:.2f})" for i, v in gap["보정후비율_M0"].tail(5)[::-1].items()))

    # ---- 확장 모델 계수 (quasi-binomial 표준오차: 과산포 반영)
    cols, b, cov, disp = last_fit
    se = np.sqrt(np.diag(cov) * disp)
    coef = pd.DataFrame({"변수": cols, "계수": b, "표준오차": se,
                         "오즈비": np.exp(b), "OR_하한": np.exp(b - 1.96 * se), "OR_상한": np.exp(b + 1.96 * se)})
    coef.to_csv(OUT_DIR / "m4_coefficients.csv", index=False, encoding="utf-8-sig")
    print("\n=== M4 지역 특성 계수 (연령·성별 보정 후 오즈비, 95% CI) ===")
    for _, r in coef[coef["변수"].str.startswith(("유형_", "고령비중"))].iterrows():
        print(f"{r['변수']:<12} OR={r['오즈비']:.3f}  [{r['OR_하한']:.3f}, {r['OR_상한']:.3f}]")
    print("(고령비중_z: 시군구 70세 이상 비중이 1SD 높을 때 / 유형: '구' 대비)")

    # ---- 그래프
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = [m.split()[0] for m in res["모델"]]
    for a, col, title in zip(ax, ["가중MAE(%p)", "평균편차", "지역잔차SD"],
                             ["교차검증 가중 MAE (%p, 낮을수록 좋음)", "교차검증 평균 편차 (낮을수록 좋음)",
                              "지역 잔차 SD (낮을수록 지역 격차 잘 설명)"]):
        bars = a.bar(labels, res[col], color=["#888"] + ["#3b78c2"] * (len(res) - 2) + ["#d9822b"])
        a.bar_label(bars, fmt="%.3f" if col == "지역잔차SD" else "%.2f", fontsize=8)
        a.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_model_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    sel = pd.concat([gap["보정후비율_M0"].head(10), gap["보정후비율_M0"].tail(10)])
    ax.barh(sel.index, sel.values - 1, left=1, color=np.where(sel.values < 1, "#3b78c2", "#d9822b"))
    ax.axvline(1, color="k", lw=0.8)
    ax.set_title("연령·성별 보정 후 이용률 격차 (실제÷기대, 상·하위 10)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_region_gap.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y / n * 100, oof_all[best] * 100, s=6, alpha=0.4)
    ax.plot([0, 100], [0, 100], "r--", lw=1)
    ax.set_xlabel("실제 이용률 (%)")
    ax.set_ylabel("예측 이용률 (%, 교차검증)")
    ax.set_title(f"실제 vs 예측 · {best}", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_pred_vs_obs.png", dpi=150)
    plt.close(fig)

    print(f"\n저장 위치: {OUT_DIR} (model_comparison.csv, region_gap.csv, m4_coefficients.csv, fig1~3.png)")


if __name__ == "__main__":
    main()
