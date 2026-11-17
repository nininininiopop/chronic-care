"""주제 5: 고령에서 나타나는 성별 이용률 역전 현상 분석

질문
    1) 남성 > 여성이던 이용률이 몇 세 구간에서 역전되는가?
    2) 역전은 지역마다 같은가? (70대 / 80세 이상)
    3) 80세 이상 성별 격차가 큰 지역의 특성은 무엇인가?

실행:
    python preprocess.py
    python sex_gap.py        # output/ 에 표·그래프 저장
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent
if BASE_DIR.name in ("scripts", "pages"):  # scripts/ 또는 pages/ 에 두어도 프로젝트 루트 기준으로 동작
    BASE_DIR = BASE_DIR.parent
PROC_DIR = BASE_DIR / "data" / "processed"
OUT_DIR = BASE_DIR / "output"
Z_CRIT = 1.96  # 지역별 검정은 다중비교 보정 없는 참고용


def setup_font() -> None:
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Malgun Gothic", "AppleGothic", "NanumGothic", "NanumBarunGothic",
                 "Noto Sans CJK KR", "Noto Sans KR"):
        if cand in names:
            plt.rcParams["font.family"] = cand
            plt.rcParams["axes.unicode_minus"] = False
            return
    print("⚠ 한글 폰트를 찾지 못했어요. 그래프의 한글이 깨질 수 있습니다. (나눔고딕 설치 권장)")


def load():
    mp, fp = PROC_DIR / "model_input.csv", PROC_DIR / "region_features.csv"
    if not mp.exists() or not fp.exists():
        sys.exit("data/processed/ 파일이 없습니다. 먼저 `python preprocess.py`를 실행하세요.")
    return pd.read_csv(mp), pd.read_csv(fp)


# ------------------------------------------------------------------ 1) 전국 연령별 성별 격차
def national_gap(m: pd.DataFrame) -> pd.DataFrame:
    g = m.groupby(["연령_수치", "성별"])[["분자", "분모"]].sum().unstack("성별")
    rows = []
    for age in g.index:
        a, nf = g.loc[age, ("분자", "여성")], g.loc[age, ("분모", "여성")]
        c, nm = g.loc[age, ("분자", "남성")], g.loc[age, ("분모", "남성")]
        pf, pm = a / nf, c / nm
        # 오즈비(여성/남성)와 95% 신뢰구간 (로그 오즈비의 정규근사)
        log_or = np.log((a / (nf - a)) / (c / (nm - c)))
        se = np.sqrt(1 / a + 1 / (nf - a) + 1 / c + 1 / (nm - c))
        rows.append({
            "연령_수치": age, "남성이용률": pm * 100, "여성이용률": pf * 100,
            "격차(여-남,%p)": (pf - pm) * 100,
            "오즈비(여/남)": np.exp(log_or),
            "OR_하한": np.exp(log_or - Z_CRIT * se), "OR_상한": np.exp(log_or + Z_CRIT * se),
        })
    out = pd.DataFrame(rows)
    out["연령"] = out["연령_수치"].map(lambda a: "80세 이상" if a == 80 else f"{a}대" if a else "0~9세")
    return out


# ------------------------------------------------------------------ 2) 지역별 성별 격차
def region_gap(m: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    p = m.pivot_table(index=["지역", "시도"], columns=["연령_수치", "성별"],
                      values=["분자", "분모"], aggfunc="sum")
    out = pd.DataFrame(index=p.index)
    for age in (70, 80):
        nf, nm = p[("분모", age, "여성")], p[("분모", age, "남성")]
        pf, pm = p[("분자", age, "여성")] / nf, p[("분자", age, "남성")] / nm
        se = np.sqrt(pf * (1 - pf) / nf + pm * (1 - pm) / nm)
        out[f"gap{age}"] = (pf - pm) * 100
        out[f"z{age}"] = (pf - pm) / se
    out["여성비중_80"] = p[("분모", 80, "여성")] / (p[("분모", 80, "여성")] + p[("분모", 80, "남성")])
    out = out.reset_index().merge(
        feat[["지역", "총인구", "고령비중_70세이상"]], on="지역", how="left", validate="one_to_one")
    last = out["지역"].str.split().str[-1].str[-1]
    out["행정유형"] = last.where(last.isin(["구", "시", "군"]), "기타")
    for age in (70, 80):
        out[f"판정{age}"] = np.select(
            [out[f"z{age}"] > Z_CRIT, out[f"z{age}"] < -Z_CRIT], ["여성 우세", "남성 우세"], "차이 불분명")
    return out


# ------------------------------------------------------------------ 3) 80세 이상 격차의 지역 특성
def ols(X: np.ndarray, y: np.ndarray, names: list[str]):
    X1 = np.column_stack([np.ones(len(X)), X])
    b = np.linalg.lstsq(X1, y, rcond=None)[0]
    res = y - X1 @ b
    dof = len(y) - X1.shape[1]
    se = np.sqrt(np.diag(res @ res / dof * np.linalg.inv(X1.T @ X1)))
    t = b / se
    table = pd.DataFrame({"변수": ["절편"] + names, "계수": b, "표준오차": se, "t": t,
                          "p": 2 * stats.t.sf(np.abs(t), dof)})
    r2 = 1 - (res @ res) / ((y - y.mean()) @ (y - y.mean()))
    return table, r2


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    setup_font()
    m, feat = load()

    # 1) 전국
    nat = national_gap(m)
    nat.to_csv(OUT_DIR / "sex_gap_by_age.csv", index=False, encoding="utf-8-sig")
    print("=== 1) 전국 연령대별 성별 이용률 ===")
    print(nat[["연령", "남성이용률", "여성이용률", "격차(여-남,%p)", "오즈비(여/남)", "OR_하한", "OR_상한"]]
          .round(2).to_string(index=False))
    sign = np.sign(nat["격차(여-남,%p)"].to_numpy())
    flips = [f"{nat['연령'][i]}→{nat['연령'][i + 1]}" for i in range(len(sign) - 1)
             if sign[i] != sign[i + 1] and sign[i] != 0]
    print(f"성별 우세가 바뀌는 구간(0%p 제외): {', '.join(flips) if flips else '없음'}")

    # 2) 지역
    reg = region_gap(m, feat)
    reg.to_csv(OUT_DIR / "region_sex_gap.csv", index=False, encoding="utf-8-sig")
    print("\n=== 2) 지역별 성별 격차 (여성-남성, %p) · 시군구 252곳 ===")
    for age in (70, 80):
        vc = reg[f"판정{age}"].value_counts()
        label = "70대" if age == 70 else "80세 이상"
        print(f"{label}: 평균 {reg[f'gap{age}'].mean():+.2f}%p · 범위 {reg[f'gap{age}'].min():+.1f}~"
              f"{reg[f'gap{age}'].max():+.1f}%p · " + ", ".join(f"{k} {v}곳" for k, v in vc.items()))
    print("70대 격차가 가장 큰 쪽(여성 우세):",
          ", ".join(f"{r.지역}({r.gap70:+.1f})" for r in reg.nlargest(3, "gap70").itertuples()))
    print("70대 격차가 가장 작은 쪽(남성 우세):",
          ", ".join(f"{r.지역}({r.gap70:+.1f})" for r in reg.nsmallest(3, "gap70").itertuples()))
    print("80세 이상 격차 큰 곳:",
          ", ".join(f"{r.지역}({r.gap80:+.1f})" for r in reg.nlargest(3, "gap80").itertuples()))
    print("80세 이상 격차 작은 곳:",
          ", ".join(f"{r.지역}({r.gap80:+.1f})" for r in reg.nsmallest(3, "gap80").itertuples()))

    # 3) 상관·회귀
    print("\n=== 3) 80세 이상 성별 격차와 지역 특성 (스피어만 상관) ===")
    for col, label in [("고령비중_70세이상", "70세 이상 인구 비중"), ("여성비중_80", "80세 이상 중 여성 비중")]:
        rho, pv = stats.spearmanr(reg[col], reg["gap80"])
        print(f"{label:<22} rho={rho:+.2f} (p={pv:.3g})")
    rho, pv = stats.spearmanr(np.log(reg["총인구"]), reg["gap80"])
    print(f"{'총인구(log)':<22} rho={rho:+.2f} (p={pv:.3g})")

    z = lambda s: ((s - s.mean()) / s.std()).to_numpy()
    X = np.column_stack([z(reg["고령비중_70세이상"]), z(reg["여성비중_80"]), z(np.log(reg["총인구"])),
                         (reg["행정유형"] == "시").astype(float), (reg["행정유형"] == "군").astype(float)])
    names = ["고령비중(z)", "여성비중_80(z)", "log인구(z)", "유형_시(구 대비)", "유형_군(구 대비)"]
    tbl, r2 = ols(X, reg["gap80"].to_numpy(), names)
    tbl.to_csv(OUT_DIR / "sex_gap_ols.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"R2": [r2], "n": [len(reg)]}).to_csv(
        OUT_DIR / "sex_gap_meta.csv", index=False, encoding="utf-8-sig")
    print(f"\nOLS (종속: 80세 이상 격차 %p, R²={r2:.2f}, n={len(reg)})")
    print(tbl.round(3).to_string(index=False))

    # 그래프
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    x = np.arange(len(nat))
    ax[0, 0].plot(x, nat["남성이용률"], "o-", label="남성", color="#3b78c2")
    ax[0, 0].plot(x, nat["여성이용률"], "o-", label="여성", color="#d9822b")
    ax[0, 0].set_xticks(x, nat["연령"], rotation=30)
    ax[0, 0].set_ylabel("이용률 (%)")
    ax[0, 0].set_title("전국 연령대별 성별 이용률")
    ax[0, 0].legend()

    bars = ax[0, 1].bar(x, nat["격차(여-남,%p)"], color=np.where(nat["격차(여-남,%p)"] < 0, "#3b78c2", "#d9822b"))
    ax[0, 1].bar_label(bars, fmt="%.1f", fontsize=8)
    ax[0, 1].axhline(0, color="k", lw=0.8)
    ax[0, 1].set_xticks(x, nat["연령"], rotation=30)
    ax[0, 1].set_title("성별 격차 (여성-남성, %p)")

    bins = np.linspace(min(reg.gap70.min(), reg.gap80.min()), max(reg.gap70.max(), reg.gap80.max()), 25)
    ax[1, 0].hist(reg["gap70"], bins=bins, alpha=0.6, label="70대", color="#3b78c2")
    ax[1, 0].hist(reg["gap80"], bins=bins, alpha=0.6, label="80세 이상", color="#d9822b")
    ax[1, 0].axvline(0, color="k", lw=0.8)
    ax[1, 0].set_xlabel("성별 격차 (여성-남성, %p)")
    ax[1, 0].set_ylabel("시군구 수")
    ax[1, 0].set_title("지역별 성별 격차 분포")
    ax[1, 0].legend()

    sc = ax[1, 1].scatter(reg["여성비중_80"] * 100, reg["gap80"], c=reg["고령비중_70세이상"] * 100,
                          cmap="viridis", s=14)
    fig.colorbar(sc, ax=ax[1, 1], label="70세 이상 인구 비중 (%)")
    ax[1, 1].set_xlabel("80세 이상 중 여성 비중 (%)")
    ax[1, 1].set_ylabel("80세 이상 성별 격차 (%p)")
    ax[1, 1].set_title("80세 이상 격차와 여성 비중")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_sex_gap.png", dpi=150)
    plt.close(fig)
    print(f"\n저장 위치: {OUT_DIR} (sex_gap_by_age.csv, region_sex_gap.csv, sex_gap_ols.csv, fig_sex_gap.png)")


if __name__ == "__main__":
    main()
