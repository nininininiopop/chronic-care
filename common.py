"""고혈압·당뇨병 의료이용률(2024) 전처리 스크립트

실행:
    python preprocess.py [CSV경로]

CSV 경로를 생략하면 이 파일과 같은 폴더(또는 data/)의 CSV를 자동으로 찾습니다.

결과물 (data/processed/):
    clean_all.csv        합계 행(전국/시도/ALL)까지 포함한 정리본 (대시보드·확인용)
    model_input.csv      모델링용. 시군구 × 성별(남/여) × 연령대, 합계 행 제외
    region_features.csv  시군구 단위 특성 (군집분석·회귀용, 연령별 이용률 wide 형태 포함)
"""

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
if BASE_DIR.name in ("scripts", "pages"):  # scripts/ 또는 pages/ 에 두어도 프로젝트 루트 기준으로 동작
    BASE_DIR = BASE_DIR.parent
OUT_DIR = BASE_DIR / "data" / "processed"

SMALL_N = 1000  # 이 값 미만이면 소표본으로 표시

GENDER_MAP = {"1": "남성", "2": "여성", "ALL": "전체"}
AGE_LABEL = {
    "00": "0~9세", "10": "10대", "20": "20대", "30": "30대", "40": "40대",
    "50": "50대", "60": "60대", "70": "70대", "80": "80세 이상", "ALL": "전체",
}


def find_csv() -> Path:
    for folder in (BASE_DIR, BASE_DIR / "data"):
        if folder.is_dir():
            files = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".csv")
            if files:
                return files[0]
    sys.exit("CSV 파일을 찾지 못했습니다. 경로를 인자로 넘겨 주세요.")


def read_csv(path: Path) -> pd.DataFrame:
    """UTF-8 → CP949 순서로 인코딩을 시도한다. 코드값('00', 'ALL')은 문자열로 유지."""
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc, dtype={"성별": str, "연령그룹": str})
        except UnicodeDecodeError:
            continue
    sys.exit("인코딩을 인식하지 못했습니다. (UTF-8 / CP949만 지원)")


def validate(df: pd.DataFrame) -> None:
    """데이터 품질 점검. 문제가 있으면 바로 멈춘다."""
    key = ["시도", "시군구", "성별", "연령그룹"]
    assert df.isna().sum().sum() == 0, "결측치가 있습니다."
    assert not df.duplicated(key).any(), "중복 행이 있습니다."
    assert (df["분자"] <= df["분모"]).all(), "분자가 분모보다 큰 행이 있습니다."

    calc = (df["분자"] / df["분모"] * 100).round(2)
    assert ((calc - df["이용률"]).abs() <= 0.01).all(), "이용률이 분자/분모와 맞지 않습니다."

    # 시군구 합 = 시도 합계 확인
    sgg = df[(df["구분"] == "시군구") & (df["성별"] != "ALL") & (df["연령그룹"] != "ALL")]
    summed = sgg.groupby("시도")[["분모", "분자"]].sum()
    total = df[
        (df["구분"] == "시도") & (df["성별"] == "ALL") & (df["연령그룹"] == "ALL")
    ].set_index("시도")[["분모", "분자"]]
    assert (summed - total.loc[summed.index]).abs().sum().sum() == 0, "시군구 합계가 시도 합계와 다릅니다."
    print("✔ 검증 통과: 결측·중복·분자>분모·이용률 재계산·시군구 합계 일치")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_csv()
    print(f"입력 파일: {path.name}")
    df = read_csv(path)

    # 1) 컬럼명 정리
    df = df.rename(columns={"분모(명)": "분모", "분자(명)": "분자", "고당의료이용률": "이용률"})

    # 2) 집계 단위 구분 (전국 / 시도 합계 / 시군구)
    df["구분"] = "시군구"
    df.loc[df["시군구"] == "전체", "구분"] = "시도"
    df.loc[df["시도"] == "전국", "구분"] = "전국"

    validate(df)

    # 3) 코드값 → 라벨, 숫자형 연령, 파생변수
    df["연령그룹_코드"] = df["연령그룹"]
    df["성별"] = df["성별"].map(GENDER_MAP)
    df["연령"] = df["연령그룹"].map(AGE_LABEL)
    df["연령_시작"] = pd.to_numeric(df["연령그룹"], errors="coerce")  # ALL → NaN
    df["미이용"] = df["분모"] - df["분자"]
    df["소표본"] = df["분모"] < SMALL_N
    df["이용0"] = df["분자"] == 0

    # 동명 시군구(중구·동구·서구 등) 구분용 고유 키
    df["지역"] = df["시도"]
    m = df["구분"] == "시군구"
    df.loc[m, "지역"] = df.loc[m, "시도"] + " " + df.loc[m, "시군구"]

    cols = ["기준연도", "구분", "시도", "시군구", "지역", "성별", "연령", "연령_시작",
            "분모", "분자", "미이용", "이용률", "소표본", "이용0"]
    clean = df[cols].copy()

    # 4) 모델 입력: 시군구 × 성별(남/여) × 연령대. 합계 행은 중복 집계라 제외
    model = clean[
        (clean["구분"] == "시군구") & (clean["성별"] != "전체") & (clean["연령"] != "전체")
    ].copy()
    model = model.rename(columns={"연령_시작": "연령_수치"})
    model["연령_수치"] = model["연령_수치"].astype(int)

    # 5) 시군구 단위 특성: 고령 인구 비중 + 연령별 이용률(wide)
    both = clean[
        (clean["구분"] == "시군구") & (clean["성별"] == "전체") & (clean["연령"] != "전체")
    ]
    age_pop = both.pivot_table(index="지역", columns="연령_시작", values="분모", aggfunc="sum")
    feat = pd.DataFrame(index=age_pop.index)
    feat["총인구"] = age_pop.sum(axis=1)
    feat["고령비중_60세이상"] = age_pop[[60, 70, 80]].sum(axis=1) / feat["총인구"]
    feat["고령비중_70세이상"] = age_pop[[70, 80]].sum(axis=1) / feat["총인구"]

    rate = both.pivot_table(index="지역", columns="연령", values="이용률")
    rate = rate[[AGE_LABEL[c] for c in ["00", "10", "20", "30", "40", "50", "60", "70", "80"]]]
    rate.columns = [f"이용률_{c}" for c in rate.columns]
    feat = feat.join(rate)

    overall = clean[
        (clean["구분"] == "시군구") & (clean["성별"] == "전체") & (clean["연령"] == "전체")
    ].set_index("지역")[["시도", "시군구", "이용률"]]
    feat = overall.rename(columns={"이용률": "이용률_전체"}).join(feat).reset_index()

    # 6) 저장 (엑셀에서도 한글이 깨지지 않도록 utf-8-sig)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(OUT_DIR / "clean_all.csv", index=False, encoding="utf-8-sig")
    model.to_csv(OUT_DIR / "model_input.csv", index=False, encoding="utf-8-sig")
    feat.to_csv(OUT_DIR / "region_features.csv", index=False, encoding="utf-8-sig")

    print(f"\nclean_all.csv        {len(clean):>6,}행")
    print(f"model_input.csv      {len(model):>6,}행  (시군구 {model['지역'].nunique()}곳 × 성별 2 × 연령 9)")
    print(f"region_features.csv  {len(feat):>6,}행")
    print(f"소표본(분모<{SMALL_N:,}) 행: {int(model['소표본'].sum())}개 · 이용 0명 행: {int(model['이용0'].sum())}개 (model_input 기준)")
    print(f"저장 위치: {OUT_DIR}")


if __name__ == "__main__":
    main()
