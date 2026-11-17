"""주제 6: 연령별 이용 패턴이 비슷한 시군구 군집화

아이디어
    지역마다 이용률의 '수준'은 연령 구성에 크게 좌우된다. 그래서 연령대별 이용률을
    전국 평균 대비 비율로 바꿔서(30대~80세 이상) '연령 프로파일'만 비교한다.
    (0~29세는 이용률이 1~2% 이하라 변동이 커서 제외)

실행:
    python preprocess.py
    python cluster_regions.py [군집수]     # 기본 3

결과: output/ 에 군집 배정표·프로파일 표·그래프 저장
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
if BASE_DIR.name in ("scripts", "pages"):  # scripts/ 또는 pages/ 에 두어도 프로젝트 루트 기준으로 동작
    BASE_DIR = BASE_DIR.parent
PROC_DIR = BASE_DIR / "data" / "processed"
OUT_DIR = BASE_DIR / "output"

SEED = 42
AGES = ["30대", "40대", "50대", "60대", "70대", "80세 이상"]
K_DEFAULT = 3
K_RANGE = range(2, 8)
N_BOOT = 100


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


def main() -> None:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else K_DEFAULT
    OUT_DIR.mkdir(exist_ok=True)
    setup_font()
    m, feat = load()

    # 1) 특성: 연령대별 이용률 ÷ 전국 같은 연령대 이용률
    nat = m.groupby("연령")[["분자", "분모"]].sum()
    nat_rate = nat["분자"] / nat["분모"] * 100
    ratio = pd.DataFrame({a: feat[f"이용률_{a}"] / nat_rate[a] for a in AGES})
    Z = StandardScaler().fit_transform(ratio)

    # 2) 군집 수 탐색
    print("=== 군집 수 탐색 (실루엣 계수: 1에 가까울수록 군집이 뚜렷) ===")
    sil = {}
    for kk in K_RANGE:
        labels = KMeans(kk, n_init=20, random_state=SEED).fit_predict(Z)
        sil[kk] = silhouette_score(Z, labels)
        print(f"k={kk}: {sil[kk]:.3f}" + ("  ← 선택" if kk == k else ""))

    km = KMeans(k, n_init=50, random_state=SEED).fit(Z)
    pca = PCA(n_components=2, random_state=SEED).fit(Z)
    pcs = pca.transform(Z)
    if np.corrcoef(pcs[:, 0], ratio.mean(axis=1))[0, 1] < 0:
        pcs[:, 0] *= -1  # PC1이 클수록 이용 비율이 높도록 방향 통일
    print(f"PCA: PC1 {pca.explained_variance_ratio_[0] * 100:.0f}%, PC2 {pca.explained_variance_ratio_[1] * 100:.0f}% 설명")

    # 군집 번호를 PC1 평균 오름차순으로 재부여
    order = pd.Series(pcs[:, 0]).groupby(km.labels_).mean().sort_values().index
    remap = {old: new for new, old in enumerate(order)}
    lab = np.array([remap[v] for v in km.labels_])
    names = ["젊은층 이용 낮음형", "평균형", "젊은층 이용 높음형"] if k == 3 else [f"유형{i + 1}" for i in range(k)]

    # 3) 안정성: 부트스트랩 재군집화 후 원래 배정과의 ARI
    rng = np.random.default_rng(SEED)
    aris = []
    for _ in range(N_BOOT):
        idx = rng.choice(len(Z), len(Z), replace=True)
        bm = KMeans(k, n_init=5, random_state=int(rng.integers(1e6))).fit(Z[idx])
        aris.append(adjusted_rand_score(lab, bm.predict(Z)))
    print(f"\n안정성(부트스트랩 {N_BOOT}회 ARI 평균): {np.mean(aris):.2f}  (1에 가까울수록 안정)")

    # 4) 군집 프로파일
    df = feat.copy()
    df["유형"] = [names[i] for i in lab]
    df["유형번호"] = lab
    df["PC1"], df["PC2"] = pcs[:, 0], pcs[:, 1]
    last = df["시군구"].str[-1]
    df["행정유형"] = last.where(last.isin(["구", "시", "군"]), "기타")
    for a in AGES:
        df[f"비율_{a}"] = ratio[a]
    # 군집 중심에서 가까운 대표 지역
    dist = np.linalg.norm(Z - km.cluster_centers_[km.labels_], axis=1)
    df["중심거리"] = dist

    prof_rows = []
    print("\n=== 군집별 프로파일 ===")
    for i in range(k):
        g = df[df["유형번호"] == i]
        top_sido = g["시도"].value_counts().head(3)
        rep = g.nsmallest(3, "중심거리")["지역"].tolist()
        shares = g["행정유형"].value_counts(normalize=True)
        row = {
            "유형": names[i], "지역수": len(g),
            "이용률_전체(평균)": g["이용률_전체"].mean(),
            "70세이상비중(평균)": g["고령비중_70세이상"].mean(),
            "총인구(중앙값)": g["총인구"].median(),
            "구%": shares.get("구", 0) * 100, "시%": shares.get("시", 0) * 100, "군%": shares.get("군", 0) * 100,
            **{f"비율_{a}": g[f"비율_{a}"].mean() for a in AGES},
            "시도 상위": ", ".join(f"{s}({c})" for s, c in top_sido.items()),
            "대표지역": ", ".join(rep),
        }
        prof_rows.append(row)
        print(f"[{names[i]}] {len(g)}곳 · 전체 이용률 {row['이용률_전체(평균)']:.1f}% · "
              f"70세이상 비중 {row['70세이상비중(평균)'] * 100:.1f}% · 구/시/군 {row['구%']:.0f}/{row['시%']:.0f}/{row['군%']:.0f}%")
        print("   전국 대비 비율:", " ".join(f"{a} {row[f'비율_{a}']:.2f}" for a in AGES))
        print("   시도 상위:", row["시도 상위"], "| 대표:", row["대표지역"])
    prof = pd.DataFrame(prof_rows)
    prof.to_csv(OUT_DIR / "cluster_profile.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"k": [k], "실루엣": [sil[k]], "부트스트랩ARI": [float(np.mean(aris))],
                  "PC1설명률": [float(pca.explained_variance_ratio_[0])]}).to_csv(
        OUT_DIR / "cluster_meta.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"k": list(sil), "실루엣": list(sil.values())}).to_csv(
        OUT_DIR / "cluster_silhouette.csv", index=False, encoding="utf-8-sig")
    df[["지역", "시도", "시군구", "유형", "PC1", "PC2", "이용률_전체", "고령비중_70세이상",
        "행정유형"] + [f"비율_{a}" for a in AGES]].to_csv(
        OUT_DIR / "cluster_assignments.csv", index=False, encoding="utf-8-sig")

    # 5) 그래프
    colors = plt.cm.tab10(np.arange(k))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    ax[0].plot(list(sil), list(sil.values()), "o-")
    ax[0].axvline(k, color="r", ls="--", lw=1)
    ax[0].set_xlabel("군집 수 k")
    ax[0].set_ylabel("실루엣 계수")
    ax[0].set_title("군집 수 탐색")
    for i in range(k):
        g = df[df["유형번호"] == i]
        ax[1].scatter(g["PC1"], g["PC2"], s=14, color=colors[i], label=f"{names[i]} ({len(g)})")
        ax[2].plot(AGES, prof.loc[i, [f"비율_{a}" for a in AGES]].astype(float), "o-", color=colors[i], label=names[i])
    ax[1].set_xlabel("PC1 (이용 비율 높을수록 →)")
    ax[1].set_ylabel("PC2")
    ax[1].set_title("시군구 분포 (PCA)")
    ax[1].legend(fontsize=8)
    ax[2].axhline(1, color="k", lw=0.8)
    ax[2].set_ylabel("전국 대비 이용률 비율")
    ax[2].set_title("군집별 연령 프로파일")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_clusters.png", dpi=150)
    plt.close(fig)
    print(f"\n저장 위치: {OUT_DIR} (cluster_assignments.csv, cluster_profile.csv, fig_clusters.png)")


if __name__ == "__main__":
    main()
