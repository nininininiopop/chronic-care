"""지역 유형 페이지

scripts/cluster_regions.py 가 미리 만들어 둔 output/ 폴더의 결과 CSV를 읽어서 보여주기만 합니다.
(무거운 계산은 하지 않아서 패키지는 requirements.txt 의 streamlit·pandas·plotly 면 충분합니다.)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 루트의 common.py 사용

import pandas as pd  # noqa: F401
import plotly.express as px
import streamlit as st

from common import load_result, missing_notice, show_chart, show_table

st.set_page_config(page_title="지역 유형", layout="wide")
st.title("지역 유형")

assign = load_result("cluster_assignments.csv")
prof = load_result("cluster_profile.csv")
meta = load_result("cluster_meta.csv")
if assign is None or prof is None:
    missing_notice("cluster_regions.py")
else:
    type_order = list(prof["유형"])
    ratio_cols = [c for c in prof.columns if c.startswith("비율_")]
    ages = [c.replace("비율_", "") for c in ratio_cols]

    st.subheader("연령별 이용 패턴으로 나눈 지역 유형")
    st.caption("연령대별 이용률을 전국 같은 연령대 이용률로 나눈 값(30대~80세 이상)을 K-means로 군집화했어요.")
    if meta is not None:
        mt = meta.iloc[0]
        st.caption(
            f"실루엣 {mt['실루엣']:.2f} · 부트스트랩 안정성(ARI) {mt['부트스트랩ARI']:.2f} · "
            f"1축이 변동의 {mt['PC1설명률'] * 100:.0f}% 설명. 경계가 뚜렷한 유형이라기보다 연속적인 분포를 나눈 것으로 보세요."
        )

    show_cols = ["유형", "지역수", "이용률_전체(평균)", "70세이상비중(평균)", "구%", "시%", "군%", "시도 상위", "대표지역"]
    show_table(prof[[c for c in show_cols if c in prof.columns]].round(3))

    left, right = st.columns(2)
    long = prof.melt(id_vars="유형", value_vars=ratio_cols, var_name="연령", value_name="전국 대비 비율")
    long["연령"] = long["연령"].str.replace("비율_", "", regex=False)
    fig = px.line(long, x="연령", y="전국 대비 비율", color="유형", markers=True,
                  category_orders={"연령": ages, "유형": type_order}, title="유형별 연령 프로파일")
    fig.add_hline(y=1, line_color="black", line_width=1)
    fig.update_layout(xaxis_title=None)
    with left:
        show_chart(fig)
    fig = px.scatter(assign, x="PC1", y="PC2", color="유형", hover_name="지역",
                     category_orders={"유형": type_order}, title="시군구 분포 (주성분 2축)")
    with right:
        show_chart(fig)

    st.subheader("지역 조회")
    pick = st.selectbox("시군구", sorted(assign["지역"]), key="cluster_region")
    row = assign[assign["지역"] == pick].iloc[0]
    st.markdown(f"**{pick}** → **{row['유형']}**  (전체 이용률 {row['이용률_전체']:.1f}%, "
                f"70세 이상 비중 {row['고령비중_70세이상'] * 100:.1f}%)")
    type_mean = prof[prof["유형"] == row["유형"]].iloc[0]
    cmp_df = pd.DataFrame({
        "연령": ages,
        "이 지역(전국 대비)": [row[c] for c in ratio_cols],
        "유형 평균(전국 대비)": [type_mean[c] for c in ratio_cols],
    })
    show_table(cmp_df.round(3))

    st.subheader("지역 목록")
    g1, g2 = st.columns(2)
    f_sido = g1.multiselect("시도 (비우면 전체)", sorted(assign["시도"].unique()), key="cluster_sido")
    f_type = g2.multiselect("유형 (비우면 전체)", type_order, key="cluster_type")
    lst = assign
    if f_sido:
        lst = lst[lst["시도"].isin(f_sido)]
    if f_type:
        lst = lst[lst["유형"].isin(f_type)]
    show_table(lst[["지역", "유형", "이용률_전체", "고령비중_70세이상", "행정유형"]].round(3))
