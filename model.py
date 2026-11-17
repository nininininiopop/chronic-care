"""분류 모델: 지역 특성으로 지역의 이용 패턴을 맞출 수 있는가

과제 A (3분류) 지역 유형: 젊은층 이용 낮음형 / 평균형 / 젊은층 이용 높음형
                          (cluster_regions.py 의 군집 결과가 정답)
과제 B (이진)  보정 후 이용 격차 방향: 기대보다 많음 / 기대보다 적음
                          (model.py 의 M0 실제÷기대 비율 > 1 이면 '많음')

입력 특성은 모두 '인구 구성·행정유형·시도'이며 이용률은 쓰지 않는다. (정답 누출 방지)
    연령대별 인구 비중(10대~80세 이상), 로그 총인구, 여성 비중, 행정유형(구/시/군), 시도

모델: 기준(최빈 클래스) / 로지스틱 회귀 / 랜덤포레스트 / 그래디언트 부스팅
평가: 5-fold × 5회 반복 층화 교차검증 (정확도, macro-F1, ROC-AUC)

실행 순서:
    python preprocess.py → python model.py → python cluster_regions.py → python classify.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
if BASE_DIR.name in ("scripts", "pages"):  # scripts/ 또는 pages/ 에 두어도 프로젝트 루트 기준으로 동작
    BASE_DIR = BASE_DIR.parent
PROC_DIR = BASE_DIR / "data" / "processed"
OUT_DIR = BASE_DIR / "output"
SEED = 42
BASELINE = "기준(최빈 클래스)"
AGE_LABEL = {10: "10대", 20: "20대", 30: "30대", 40: "40대", 50: "50대", 60: "60대", 70: "70대", 80: "80세 이상"}


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
    needed = [PROC_DIR / "model_input.csv", PROC_DIR / "region_features.csv",
              OUT_DIR / "cluster_assignments.csv", OUT_DIR / "region_gap.csv"]
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        sys.exit(f"필요한 파일이 없습니다: {', '.join(missing)}\n"
                 "preprocess.py → model.py → cluster_regions.py 를 먼저 실행하세요.")
    m = pd.read_csv(needed[0])
    feat = pd.read_csv(needed[1])
    assign = pd.read_csv(needed[2]).set_index("지역")["유형"]
    gap = pd.read_csv(needed[3]).set_index("지역")["보정후비율_M0"]
    return m, feat, assign, gap


def build_features(m: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    """지역 단위 설명변수. 이용률은 포함하지 않는다."""
    pop = m.groupby(["지역", "연령_수치"])["분모"].sum().unstack()
    total = pop.sum(axis=1)
    X = pd.DataFrame(index=pop.index)
    for age, label in AGE_LABEL.items():  # 0~9세는 기준(다른 비중의 합으로 결정)이라 제외
        X[f"인구비중_{label}"] = pop[age] / total
    X["로그총인구"] = np.log(total)
    female = m[m["성별"] == "여성"].groupby("지역")["분모"].sum()
    X["여성비중"] = female / total

    info = feat.set_index("지역")[["시도", "시군구"]].reindex(X.index)
    last = info["시군구"].str[-1]
    kind = last.where(last.isin(["구", "시", "군"]), "기타")
    X = pd.concat([
        X,
        pd.get_dummies(kind, prefix="유형", drop_first=True, dtype=float),
        pd.get_dummies(info["시도"], prefix="시도", drop_first=True, dtype=float),
    ], axis=1)
    return X


def make_models() -> dict:
    return {
        BASELINE: DummyClassifier(strategy="most_frequent"),
        "로지스틱 회귀": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0)),
        "랜덤포레스트": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                          random_state=SEED, n_jobs=-1),
        "그래디언트 부스팅": GradientBoostingClassifier(random_state=SEED),
    }


def auc_score(y_true, proba, classes) -> float:
    if len(classes) == 2:
        return roc_auc_score(y_true, proba[:, 1])
    return roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=classes)


def evaluate(X: pd.DataFrame, y: np.ndarray, task: str) -> pd.DataFrame:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    rows = []
    for name, model in make_models().items():
        accs, f1s, aucs = [], [], []
        for tr, te in cv.split(X, y):
            fit = clone(model).fit(X.iloc[tr], y[tr])
            pred = fit.predict(X.iloc[te])
            accs.append(accuracy_score(y[te], pred))
            f1s.append(f1_score(y[te], pred, average="macro"))
            aucs.append(0.5 if name == BASELINE else auc_score(y[te], fit.predict_proba(X.iloc[te]), fit.classes_))
        rows.append({"과제": task, "모델": name,
                     "정확도": np.mean(accs), "정확도_SD": np.std(accs),
                     "macroF1": np.mean(f1s), "macroF1_SD": np.std(f1s),
                     "AUC": np.mean(aucs), "AUC_SD": np.std(aucs)})
    return pd.DataFrame(rows)


def importance_table(X: pd.DataFrame, y: np.ndarray, task: str) -> pd.DataFrame:
    """랜덤포레스트 중요도. 더미 변수(시도·유형)는 한 변수로 합친다."""
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, random_state=SEED, n_jobs=-1).fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=X.columns)
    group = imp.groupby(lambda c: "시도" if c.startswith("시도_") else "행정유형" if c.startswith("유형_") else c).sum()
    out = group.sort_values(ascending=False).rename("중요도").reset_index().rename(columns={"index": "변수"})
    out.insert(0, "과제", task)
    return out


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    setup_font()
    m, feat, assign, gap = load()
    X = build_features(m, feat)
    tasks = {
        "A. 지역 유형(3분류)": assign.reindex(X.index).to_numpy(),
        "B. 보정 후 이용 격차 방향(이진)": np.where(gap.reindex(X.index) > 1, "기대보다 많음", "기대보다 적음"),
    }
    print(f"데이터: 시군구 {len(X)}곳 · 특성 {X.shape[1]}개 (이용률은 사용하지 않음)\n")

    res_all, cm_all, imp_all, pred_all, meta = [], [], [], [], []
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    for i, (task, y) in enumerate(tasks.items()):
        dist = pd.Series(y).value_counts()
        print(f"=== {task} · 클래스 분포: " + ", ".join(f"{k} {v}" for k, v in dist.items()))
        res = evaluate(X, y, task)
        res_all.append(res)
        print(res.drop(columns="과제").round(3).to_string(index=False))

        best = res[res["모델"] != BASELINE].sort_values("macroF1", ascending=False).iloc[0]["모델"]
        model = make_models()[best]
        classes = np.array(sorted(set(y)))
        proba = cross_val_predict(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                  method="predict_proba")
        pred = classes[proba.argmax(axis=1)]
        cm = confusion_matrix(y, pred, labels=classes)
        print(f"→ 최고 모델(macro-F1 기준): {best} · 교차예측 정확도 {accuracy_score(y, pred):.3f}")
        print("혼동행렬(행=실제, 열=예측):")
        print(pd.DataFrame(cm, index=classes, columns=classes).to_string(), "\n")

        for a, ca in enumerate(classes):
            for b, cb in enumerate(classes):
                cm_all.append({"과제": task, "모델": best, "실제": ca, "예측": cb, "건수": int(cm[a, b])})
        imp = importance_table(X, y, task)
        imp_all.append(imp)
        pred_all.append(pd.DataFrame({
            "과제": task, "지역": X.index, "시도": feat.set_index("지역").reindex(X.index)["시도"].to_numpy(),
            "실제": y, "예측": pred, "예측확률": proba.max(axis=1), "정답": y == pred}))
        meta.append({"과제": task, "최고모델": best, "지역수": len(X),
                     "클래스분포": ", ".join(f"{k} {v}" for k, v in dist.items())})

        im = ax[i, 0].imshow(cm, cmap="Blues")
        ax[i, 0].set_xticks(range(len(classes)), classes, rotation=20, fontsize=8)
        ax[i, 0].set_yticks(range(len(classes)), classes, fontsize=8)
        for a in range(len(classes)):
            for b in range(len(classes)):
                ax[i, 0].text(b, a, cm[a, b], ha="center", va="center",
                              color="white" if cm[a, b] > cm.max() / 2 else "black")
        ax[i, 0].set_xlabel("예측")
        ax[i, 0].set_ylabel("실제")
        ax[i, 0].set_title(f"{task}\n혼동행렬 ({best}, 교차예측)", fontsize=10)
        top = imp.head(8).iloc[::-1]
        ax[i, 1].barh(top["변수"], top["중요도"], color="#3b78c2")
        ax[i, 1].set_title("랜덤포레스트 변수 중요도 (상위 8)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_classify.png", dpi=150)
    plt.close(fig)

    pd.concat(res_all).to_csv(OUT_DIR / "clf_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cm_all).to_csv(OUT_DIR / "clf_confusion.csv", index=False, encoding="utf-8-sig")
    pd.concat(imp_all).to_csv(OUT_DIR / "clf_importance.csv", index=False, encoding="utf-8-sig")
    pd.concat(pred_all).to_csv(OUT_DIR / "clf_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(meta).to_csv(OUT_DIR / "clf_meta.csv", index=False, encoding="utf-8-sig")
    print(f"저장 위치: {OUT_DIR} (clf_results/confusion/importance/predictions/meta.csv, fig_classify.png)")


if __name__ == "__main__":
    main()
