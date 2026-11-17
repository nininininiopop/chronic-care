"""분류 모델 페이지

scripts/classify.py 가 미리 만들어 둔 output/ 폴더의 결과 CSV를 읽어서 보여주기만 합니다.
(무거운 계산은 하지 않아서 패키지는 requirements.txt 의 streamlit·pandas·plotly 면 충분합니다.)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 루트의 common.py 사용

import pandas as pd  # noqa: F401
import plotly.express as px
import streamlit as st

from common import load_result, missing_notice, show_chart, show_table

st.set_page_config(page_title="분류 모델", layout="wide")
st.title("분류 모델")

clf_res = load_result("clf_results.csv")
clf_cm = load_result("clf_confusion.csv")
clf_imp = load_result("clf_importance.csv")
clf_pred = load_result("clf_predictions.csv")
clf_meta = load_result("clf_meta.csv")
if clf_res is None:
    missing_notice("classify.py")
else:
    BASE_NAME = "기준(최빈 클래스)"
    st.subheader("분류 모델: 지역 특성으로 이용 패턴 맞추기")
    st.caption(
        "입력은 연령대별 인구 비중, 로그 총인구, 여성 비중, 행정유형(구·시·군), 시도이고 이용률은 쓰지 않았어요(정답 누출 방지). "
        "5-fold × 5회 반복 층화 교차검증 결과입니다."
    )
    task = st.radio("분류 과제", list(clf_res["과제"].unique()), horizontal=True)
    r = clf_res[clf_res["과제"] == task].copy()
    base = r[r["모델"] == BASE_NAME].iloc[0]
    best = r[r["모델"] != BASE_NAME].sort_values("macroF1", ascending=False).iloc[0]

    if clf_meta is not None:
        mt = clf_meta[clf_meta["과제"] == task]
        if len(mt):
            st.caption(f"시군구 {int(mt['지역수'].iloc[0])}곳 · 클래스 분포: {mt['클래스분포'].iloc[0]}")

    st.markdown(f"**최고 모델: {best['모델']}** (기준 모델 대비)")
    k1, k2, k3 = st.columns(3)
    k1.metric("정확도", f"{best['정확도']:.3f}", f"{best['정확도'] - base['정확도']:+.3f}")
    k2.metric("macro-F1", f"{best['macroF1']:.3f}", f"{best['macroF1'] - base['macroF1']:+.3f}")
    k3.metric("AUC", f"{best['AUC']:.3f}", f"{best['AUC'] - base['AUC']:+.3f}")

    metric = st.selectbox("평가 지표", ["macroF1", "정확도", "AUC"], key="clf_metric")
    fig = px.bar(r, x="모델", y=metric, error_y=f"{metric}_SD", text=metric,
                 title=f"{metric} (막대 = 반복 교차검증 평균, 오차막대 = 표준편차)")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(xaxis_title=None)
    show_chart(fig)
    show_table(r.drop(columns="과제").round(3))

    left, right = st.columns(2)
    if clf_cm is not None:
        c = clf_cm[clf_cm["과제"] == task]
        if len(c):
            piv = c.pivot(index="실제", columns="예측", values="건수")
            fig = px.imshow(piv, text_auto=True, color_continuous_scale="Blues",
                            labels=dict(x="예측", y="실제", color="건수"),
                            title=f"혼동행렬 ({c['모델'].iloc[0]}, 교차예측)")
            with left:
                show_chart(fig)
    if clf_imp is not None:
        t = clf_imp[clf_imp["과제"] == task].head(10)
        fig = px.bar(t, x="중요도", y="변수", orientation="h", title="변수 중요도 (랜덤포레스트, 상위 10)")
        fig.update_layout(yaxis=dict(autorange="reversed", title=None))
        with right:
            show_chart(fig)
    st.caption(
        "변수 중요도는 랜덤포레스트 기준이라 최고 모델과 다를 수 있어요. "
        "집계 데이터에서 본 지역 단위 관련성이며 인과관계가 아닙니다."
    )

    if clf_pred is not None:
        st.subheader("지역별 예측 결과")
        pt = clf_pred[clf_pred["과제"] == task]
        only_wrong = st.checkbox("오분류만 보기", key="clf_wrong")
        sido_pick = st.multiselect("시도 (비우면 전체)", sorted(pt["시도"].unique()), key="clf_sido")
        if only_wrong:
            pt = pt[~pt["정답"]]
        if sido_pick:
            pt = pt[pt["시도"].isin(sido_pick)]
        st.caption(f"{len(pt)}곳 · 예측확률은 교차예측에서 예측 클래스에 부여된 확률입니다.")
        show_table(pt[["지역", "시도", "실제", "예측", "예측확률", "정답"]].round(3))
