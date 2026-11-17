"""성별 격차 페이지

scripts/sex_gap.py 가 미리 만들어 둔 output/ 폴더의 결과 CSV를 읽어서 보여주기만 합니다.
(무거운 계산은 하지 않아서 패키지는 requirements.txt 의 streamlit·pandas·plotly 면 충분합니다.)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 루트의 common.py 사용

import pandas as pd  # noqa: F401
import plotly.express as px
import streamlit as st

from common import load_result, missing_notice, show_chart, show_table

st.set_page_config(page_title="성별 격차", layout="wide")
st.title("성별 격차")

nat = load_result("sex_gap_by_age.csv")
reg = load_result("region_sex_gap.csv")
if nat is None or reg is None:
    missing_notice("sex_gap.py")
else:
    nat = nat.sort_values("연령_수치")
    age_labels = list(nat["연령"])
    st.subheader("전국 연령대별 성별 이용률")
    last = nat.iloc[-1]
    m1, m2, m3 = st.columns(3)
    m1.metric("80세 이상 성별 격차 (여-남)", f"{last['격차(여-남,%p)']:+.1f}%p")
    m2.metric("80세 이상 오즈비 (여/남)", f"{last['오즈비(여/남)']:.2f}")
    peak = nat.loc[nat["격차(여-남,%p)"].idxmin()]
    m3.metric(f"남성 우위 최대 구간 ({peak['연령']})", f"{peak['격차(여-남,%p)']:+.1f}%p")

    left, right = st.columns(2)
    long = nat.melt(id_vars="연령", value_vars=["남성이용률", "여성이용률"],
                    var_name="성별", value_name="이용률(%)")
    long["성별"] = long["성별"].str.replace("이용률", "", regex=False)
    fig = px.line(long, x="연령", y="이용률(%)", color="성별", markers=True,
                  category_orders={"연령": age_labels}, title="성별 이용률")
    fig.update_layout(xaxis_title=None)
    with left:
        show_chart(fig)
    fig = px.bar(nat, x="연령", y="격차(여-남,%p)", text="격차(여-남,%p)",
                 category_orders={"연령": age_labels}, title="성별 격차 (여성-남성, %p)")
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(xaxis_title=None)
    with right:
        show_chart(fig)
    st.caption("남성이 높다가 70대에서 거의 같아지고, 80세 이상에서 여성이 더 높아집니다.")

    st.subheader("지역별 성별 격차")
    age_opt = st.radio("연령대", ["80세 이상", "70대"], horizontal=True)
    a = "80" if age_opt == "80세 이상" else "70"
    gcol, jcol = f"gap{a}", f"판정{a}"
    counts = reg[jcol].value_counts()
    k1, k2, k3 = st.columns(3)
    k1.metric("여성 우세 (유의)", f"{int(counts.get('여성 우세', 0))}곳")
    k2.metric("남성 우세 (유의)", f"{int(counts.get('남성 우세', 0))}곳")
    k3.metric("차이 불분명", f"{int(counts.get('차이 불분명', 0))}곳")

    fig = px.histogram(reg, x=gcol, nbins=30, color=jcol,
                       labels={gcol: "성별 격차 (여성-남성, %p)", jcol: "판정"},
                       title=f"{age_opt} 지역별 성별 격차 분포")
    fig.add_vline(x=0, line_color="black")
    fig.update_layout(yaxis_title="시군구 수")
    show_chart(fig)
    st.caption("판정은 지역별 두 비율 차이의 z 검정(|z|>1.96)이며, 252곳 다중비교 보정은 하지 않은 참고용입니다.")

    view = reg.copy()
    view["80세 이상 중 여성 비중(%)"] = view["여성비중_80"] * 100
    view["70세 이상 인구 비중(%)"] = view["고령비중_70세이상"] * 100
    fig = px.scatter(view, x="80세 이상 중 여성 비중(%)", y="gap80", color="70세 이상 인구 비중(%)",
                     hover_name="지역", labels={"gap80": "80세 이상 성별 격차 (%p)"},
                     title="80세 이상 격차와 여성 비중")
    show_chart(fig)

    pick = st.selectbox("지역 조회", sorted(reg["지역"]), key="sex_region")
    row = reg[reg["지역"] == pick].iloc[0]
    r1, r2 = st.columns(2)
    r1.metric("70대 격차 (여-남)", f"{row['gap70']:+.1f}%p", row["판정70"], delta_color="off")
    r2.metric("80세 이상 격차 (여-남)", f"{row['gap80']:+.1f}%p", row["판정80"], delta_color="off")

    ols = load_result("sex_gap_ols.csv")
    meta = load_result("sex_gap_meta.csv")
    if ols is not None:
        with st.expander("80세 이상 격차와 지역 특성 (회귀)"):
            if meta is not None:
                st.caption(f"R² = {float(meta['R2'].iloc[0]):.2f}, 시군구 {int(meta['n'].iloc[0])}곳. "
                           "지역 특성으로 설명되는 부분이 작아요. 역전은 지역과 무관하게 나타나는 구조적 현상에 가깝습니다.")
            show_table(ols.round(3))
