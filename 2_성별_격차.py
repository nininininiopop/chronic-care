"""모델 비교 페이지

scripts/model.py 가 미리 만들어 둔 output/ 폴더의 결과 CSV를 읽어서 보여주기만 합니다.
(무거운 계산은 하지 않아서 패키지는 requirements.txt 의 streamlit·pandas·plotly 면 충분합니다.)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 루트의 common.py 사용

import pandas as pd  # noqa: F401
import plotly.express as px
import streamlit as st

from common import load_result, missing_notice, show_chart, show_table

st.set_page_config(page_title="모델 비교", layout="wide")
st.title("모델 비교")

res = load_result("model_comparison.csv")
if res is None:
    missing_notice("model.py")
else:
    st.subheader("기준 모델 vs 확장 모델")
    st.caption(
        "시군구 단위 5-fold 교차검증 결과예요. 같은 시군구가 학습과 평가에 섞이지 않도록 나눴습니다."
    )
    res = res.copy()
    res["코드"] = res["모델"].str.split().str[0]
    metric_info = {
        "가중MAE(%p)": "낮을수록 좋음",
        "MAE(%p)": "낮을수록 좋음",
        "RMSE(%p)": "낮을수록 좋음",
        "소표본MAE(%p)": "낮을수록 좋음 (분모 1,000명 미만 행)",
        "평균편차": "낮을수록 좋음",
        "지역잔차SD": "낮을수록 지역 격차를 잘 설명",
        "편차감소율(%)": "기준 모델 대비, 클수록 좋음",
    }
    metric = st.selectbox("평가 지표", list(metric_info), index=0)
    fig = px.bar(
        res,
        x="코드",
        y=metric,
        text=metric,
        hover_name="모델",
        title=f"{metric} · {metric_info[metric]}",
    )
    fig.update_traces(
        texttemplate="%{text:.3f}" if metric == "지역잔차SD" else "%{text:.2f}",
        textposition="outside",
    )
    fig.update_layout(xaxis_title=None)
    show_chart(fig)
    show_table(res.drop(columns="코드").round(3))

    with st.expander("모델 설명"):
        st.markdown(
            "- **M0 기준**: 연령 + 성별\n"
            "- **M1**: M0 + 연령×성별 교호작용\n"
            "- **M2**: M1 + 시도\n"
            "- **M3**: M1 + 행정유형(구·시·군) + 70세 이상 고령 비중\n"
            "- **M4 확장**: M1 + 시도 + 행정유형 + 고령 비중\n"
            "- **RF**: 랜덤포레스트 (비교용)\n\n"
            "모두 반응변수는 분자/분모(이항 모델)입니다."
        )

    gap = load_result("region_gap.csv")
    if gap is not None:
        st.subheader("연령·성별 보정 후 남는 지역 격차")
        st.caption(
            "실제 이용자 ÷ 기대 이용자(연령·성별만 고려한 기준 모델 M0). "
            "음수면 기대보다 이용이 적은 지역, 양수면 많은 지역입니다."
        )
        n_show = st.slider("상·하위 각각 표시 개수", 5, 20, 10, key="gap_n")
        gap_sorted = gap.sort_values("보정후비율_M0")
        sel = pd.concat([gap_sorted.head(n_show), gap_sorted.tail(n_show)]).copy()
        sel["기대 대비(%)"] = (sel["보정후비율_M0"] - 1) * 100
        sel["방향"] = sel["기대 대비(%)"].lt(0).map({True: "기대보다 적음", False: "기대보다 많음"})
        fig = px.bar(
            sel,
            x="기대 대비(%)",
            y="지역",
            color="방향",
            orientation="h",
            title="보정 후 이용 격차 (실제 이용자 ÷ 기대 이용자 − 1)",
        )
        fig.update_layout(yaxis=dict(autorange="reversed", title=None), height=520)
        show_chart(fig)
        with st.expander("전체 지역 표 보기"):
            show_table(gap.round(3))

    coef = load_result("m4_coefficients.csv")
    if coef is not None:
        st.subheader("지역 특성의 효과 (연령·성별·시도 보정 후)")
        sub = coef[coef["변수"].str.startswith(("유형_", "고령비중"))]
        show_table(sub.round(3))
        st.caption(
            "오즈비 1보다 크면 이용 오즈가 높다는 뜻이에요. 유형은 '구' 대비이고, "
            "고령비중_z는 70세 이상 비중이 표준편차 1만큼 높을 때입니다. "
            "신뢰구간은 과산포를 반영한 보정 표준오차로 계산했어요. 집계 데이터라 인과가 아닌 관련성으로 해석해야 합니다."
        )
