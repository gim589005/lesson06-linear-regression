import subprocess
import sys

import numpy as np
import pandas as pd
import streamlit as st

# plotly가 설치되어 있지 않으면(requirements.txt가 적용되지 않은 경우) 자동으로 설치
try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
    import plotly.graph_objects as go

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
LAST_YEAR = 2025      # 수업 기준 기간의 마지막 해
MIN_DAYS = 300        # 연도별 최소 관측일 수
SLIDER_MIN, SLIDER_MAX = 1900, 2100

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="wide")


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_yearly_mean(url: str) -> pd.DataFrame:
    """서울 일별 기온 -> 연도별 평균기온(관측일 수 포함)."""
    df = pd.read_csv(url, encoding="utf-8")
    df.columns = [str(c).strip() for c in df.columns]

    df["날짜"] = pd.to_datetime(df["날짜"].astype(str).str.strip(), errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 날짜가 없는 행 제거, 기준 기간(2025년까지)만 사용
    df = df.dropna(subset=["날짜"])
    df["연도"] = df["날짜"].dt.year
    df = df[df["연도"] <= LAST_YEAR]

    # 연도별 평균기온과 관측일 수(평균기온이 기록된 날)
    yearly = (
        df.groupby("연도")["평균기온"]
        .agg(연평균기온="mean", 관측일수="count")
        .reset_index()
    )
    # 관측일이 300일 미만인 해 제외
    yearly = yearly[yearly["관측일수"] >= MIN_DAYS].reset_index(drop=True)
    return yearly


st.title("🌡️ 기온 예측기")
st.caption("서울 연평균기온 추세(선형 회귀)로 원하는 해의 기온을 예측합니다.")

try:
    yearly = load_yearly_mean(DATA_URL)
except Exception as e:  # 네트워크/형식 오류 안내
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

if len(yearly) < 2:
    st.warning("회귀 직선을 만들기에 데이터가 부족합니다.")
    st.stop()

x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)

slope, intercept = np.polyfit(x, y, 1)
r = float(np.corrcoef(x, y)[0, 1])

n_years = len(yearly)
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())

# 최근 20년(끝 연도 포함 20개 연도 구간) 회귀
RECENT_YEARS = 20
recent = yearly[yearly["연도"] > end_year - RECENT_YEARS]
has_recent = len(recent) >= 3
if has_recent:
    xr = recent["연도"].to_numpy(dtype=float)
    yr = recent["연평균기온"].to_numpy(dtype=float)
    slope_recent, _ = np.polyfit(xr, yr, 1)
    r_recent = float(np.corrcoef(xr, yr)[0, 1])
    recent_start = int(recent["연도"].min())
    recent_n = len(recent)

per100_all = slope * 100

# --- 슬라이더와 예측값 ---
year = st.slider("예측할 연도", SLIDER_MIN, SLIDER_MAX, 2030, step=1)
pred = slope * year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding:1.2rem 0;">
        <div style="font-size:1.1rem; opacity:0.7;">{year}년 예상 연평균기온</div>
        <div style="font-size:4.5rem; font-weight:800; line-height:1.1;">{pred:.2f} ℃</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- 100년당 기온 변화 (기울기 × 100) ---
st.markdown(
    f"""
    <div style="text-align:center; padding:0.8rem 0 0.2rem 0;">
        <div style="font-size:1.1rem; opacity:0.7;">
            회귀 직선 기울기: 100년에 오르는 기온 ({start_year}~{end_year}년 전체 기간)
        </div>
        <div style="font-size:4.5rem; font-weight:800; line-height:1.1;">
            {per100_all:+.2f} ℃ / 100년
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- 회귀 직선 정보 (전체 기간) ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("회귀 직선에 사용한 해의 개수", f"{n_years}개")
c2.metric("시작 연도", f"{start_year}년")
c3.metric("끝 연도", f"{end_year}년")
c4.metric("상관계수 (r)", f"{r:.3f}")

# --- 전체 기간 vs 최근 20년 비교 ---
st.subheader("기울기 비교: 전체 기간 vs 최근 20년")
col_all, col_recent = st.columns(2)

with col_all:
    st.markdown("#### 전체 기간")
    st.markdown(
        f"<div style='font-size:2.8rem; font-weight:700;'>{per100_all:+.2f} ℃ / 100년</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"{start_year}~{end_year}년 · {n_years}개 해 · 상관계수 r = {r:.3f} · "
        f"기울기 {slope:+.4f} ℃/년"
    )

with col_recent:
    st.markdown(f"#### 최근 {RECENT_YEARS}년")
    if has_recent:
        per100_recent = slope_recent * 100
        st.markdown(
            f"<div style='font-size:2.8rem; font-weight:700;'>{per100_recent:+.2f} ℃ / 100년</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"{recent_start}~{end_year}년 · {recent_n}개 해 · 상관계수 r = {r_recent:.3f} · "
            f"기울기 {slope_recent:+.4f} ℃/년"
        )
    else:
        st.info("최근 20년 구간의 데이터가 부족해 계산하지 못했습니다.")

if has_recent:
    diff = per100_recent - per100_all
    ratio_txt = ""
    if per100_all != 0:
        ratio_txt = f" (전체 기간의 약 {per100_recent / per100_all:.1f}배)"
    st.write(
        f"최근 {RECENT_YEARS}년 기준으로 환산하면 100년에 **{per100_recent:+.2f} ℃**로, "
        f"전체 기간보다 **{diff:+.2f} ℃**만큼 차이가 납니다{ratio_txt}. "
        f"다만 {recent_n}개 해만으로 구한 기울기라 연도별 변동의 영향을 크게 받습니다."
    )

st.write(
    f"회귀식(전체 기간): 연평균기온 = **{slope:.4f}** × 연도 + **{intercept:.2f}**"
)

# --- Plotly 그래프 ---
line_x = np.array([SLIDER_MIN, SLIDER_MAX])
line_y = slope * line_x + intercept

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="연평균기온(관측)",
        marker=dict(size=8),
        hovertemplate="%{x}년<br>%{y:.2f} ℃<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="회귀 직선",
        line=dict(width=3, dash="dash"),
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter(
        x=[year],
        y=[pred],
        mode="markers",
        name=f"{year}년 예측",
        marker=dict(size=16, symbol="star", line=dict(width=1, color="white")),
        hovertemplate="%{x}년 예측<br>%{y:.2f} ℃<extra></extra>",
    )
)
fig.update_layout(
    title=f"서울 연평균기온 산점도와 회귀 직선 (r = {r:.3f})",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    xaxis=dict(range=[SLIDER_MIN, SLIDER_MAX]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    height=520,
    margin=dict(t=90),
)
st.plotly_chart(fig, use_container_width=True)

if year < start_year or year > end_year:
    st.info(
        f"{year}년은 직선을 만든 기간({start_year}~{end_year}년) 밖이라 "
        "외삽(extrapolation)한 값입니다. 참고용으로만 보세요."
    )

with st.expander("연도별 평균기온 데이터 보기"):
    st.dataframe(
        yearly.rename(columns={"연평균기온": "연평균기온(℃)"}).round(2),
        use_container_width=True,
        hide_index=True,
    )
