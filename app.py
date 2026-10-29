# -*- coding: utf-8 -*-
"""전체종목분석 Streamlit UI. 핵심 모델/스크리닝 계산은 stock_screener_v2_0.py를 사용."""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="전체종목 분석 · Stock Glass", page_icon="📊", layout="wide")

CSS = r"""
<style>
:root { color-scheme: dark; }
.stApp {
 background:
  radial-gradient(ellipse at 12% 0%, rgba(68,119,220,.25), transparent 34%),
  radial-gradient(ellipse at 92% 10%, rgba(139,92,246,.20), transparent 30%),
  linear-gradient(145deg,#070C19 0%,#0B1530 52%,#10152B 100%);
 color:#F3F6FF;
}
.block-container {max-width:1450px; padding-top:1.5rem; padding-bottom:3rem;}
.hero {position:relative; overflow:hidden; padding:clamp(1.2rem,3vw,2.1rem); border-radius:26px;
 background:linear-gradient(125deg,rgba(30,52,94,.72),rgba(18,28,56,.52));
 border:1px solid rgba(210,229,255,.23); box-shadow:0 20px 55px rgba(0,0,0,.23); backdrop-filter:blur(18px);}
.hero:after {content:""; position:absolute; width:250px;height:250px;right:-70px;top:-110px;border-radius:50%;
 background:radial-gradient(circle,rgba(118,228,247,.25),transparent 70%);pointer-events:none;}
.eyebrow {font-size:.76rem;letter-spacing:.16em;text-transform:uppercase;color:#92EAF7;font-weight:800;}
.hero h1 {font-size:clamp(1.8rem,4.2vw,3rem);line-height:1.15;margin:.55rem 0 .7rem;color:#fff;font-weight:850;}
.hero p {color:#D0DDF7;font-size:1rem;margin:0;max-width:850px;}
.glass-card {background:linear-gradient(145deg,rgba(255,255,255,.105),rgba(255,255,255,.035));border:1px solid rgba(220,234,255,.17);
 border-radius:20px;padding:1.1rem 1.2rem;box-shadow:0 14px 36px rgba(0,0,0,.14);backdrop-filter:blur(14px);}
.section-title {font-size:1.12rem;font-weight:800;color:#F4F7FF;margin:1.35rem 0 .75rem;}
.kpi-label {color:#B9C8E8;font-size:.83rem;font-weight:700;}
.kpi-value {color:#fff;font-size:clamp(1.35rem,2.5vw,2rem);font-weight:850;line-height:1.25;margin:.3rem 0;}
.kpi-sub {color:#B7C8E8;font-size:.78rem;}
.small-note {font-size:.82rem;color:#B9C8E8;}
.stMarkdown,.stMarkdown p,.stMarkdown li,.stCaption,.stText, label {color:#E5ECFF;}
[data-testid="stForm"],[data-testid="stExpander"],[data-testid="stVerticalBlockBorderWrapper"] {border-color:rgba(220,234,255,.18)!important;}
[data-testid="stMetric"] {background:rgba(255,255,255,.075);border:1px solid rgba(220,234,255,.15);border-radius:16px;padding:1rem;}
[data-testid="stMetricLabel"] {color:#C4D2EE!important;}
[data-testid="stMetricValue"] {color:#FFFFFF!important;}
[data-testid="stDataFrame"] {border:1px solid rgba(220,234,255,.18);border-radius:14px;overflow:hidden;}
[data-testid="stAlert"] {border-radius:14px;}
hr {border-color:rgba(220,234,255,.15);}
footer {visibility:hidden;}
@media(max-width:700px){.block-container{padding:.8rem .65rem 3rem}.hero{border-radius:19px;padding:1.1rem}.glass-card{padding:.9rem;border-radius:16px}.section-title{font-size:1rem}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="eyebrow">STOCK GLASS · MARKET-WIDE QUANT SCREENER</div>
  <h1>전체종목 분석 <span style="background:linear-gradient(90deg,#76E4F7,#B9A5FF);-webkit-background-clip:text;background-clip:text;color:transparent">v2.0</span></h1>
  <p>한국 KOSPI 또는 미국 S&amp;P 500을 종목별로 검증하고, 홀드아웃 성과와 독립 WFO 결과를 바탕으로 매수검토·관찰·매수제외 그룹을 분류합니다.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_engine():
    path = Path(__file__).with_name("stock_screener_v2_0.py")
    spec = importlib.util.spec_from_file_location("stock_screener_engine", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("stock_screener_v2_0.py를 불러올 수 없습니다.")
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

@st.cache_resource(show_spinner=False)
def get_core():
    return get_engine().load_core_module()


def dedupe_rows(rows, keys):
    seen, out = set(), []
    for row in rows:
        key = tuple(str(row.get(k, "")) for k in keys)
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def empty_records(engine, checkpoint_path, config):
    return engine._load_checkpoint_records(checkpoint_path, config)

with st.expander("분석 설정", expanded=True):
    with st.form("market_scan_form"):
        c1, c2 = st.columns(2)
        with c1:
            market = st.selectbox("분석 시장", ["KOSPI", "S&P 500"], index=0)
            full_universe = st.checkbox("선택 시장의 전체 구성종목 분석", value=False,
                help="전체 분석은 수백~수천 종목을 처리할 수 있어 상당한 시간이 걸립니다. 첫 실행은 일부 종목으로 연결을 확인하는 편이 안전합니다.")
            sample_count = st.number_input("시험 실행 시 분석 종목 수", min_value=1, max_value=5000, value=30, step=10,
                disabled=full_universe, help="전체 분석을 선택하지 않았을 때 적용됩니다.")
        with c2:
            target_pct = st.number_input("목표 상승률 (%)", min_value=0.1, max_value=500.0, value=15.0, step=0.5)
            max_holding = st.number_input("최대 보유 거래일", min_value=1, max_value=252, value=20, step=1)
            fee_pct = st.number_input("수수료 (% / 편도)", min_value=0.0, max_value=10.0, value=0.05, step=0.01, format="%.3f")
            slippage_pct = st.number_input("슬리피지 (% / 편도)", min_value=0.0, max_value=10.0, value=0.15, step=0.01, format="%.3f")
        run_scan = st.form_submit_button("전체종목 분석 시작", type="primary", width="stretch")

st.markdown("""
<div class="glass-card small-note">
<b>검증 기준:</b> 목표수익률 +15% · 기본 최대 보유 20거래일 · 매매 비용 차감 · 종목별 홀드아웃 및 워크포워드 검증.<br>
<b>분류:</b> 1차 기준 통과 후 2차 검증까지 통과하면 ‘매수검토’, 1차만 통과하면 ‘관찰’, 1차 미통과는 ‘매수제외’. 모델 점수는 보정된 성공확률이 아닙니다.
</div>
""", unsafe_allow_html=True)

if run_scan:
    try:
        engine = get_engine()
        with st.spinner("분석 엔진과 원본 모델을 준비하고 있습니다..."):
            core = get_core()
            universe, universe_source = engine.load_stock_universe(core, market)
        if universe.empty:
            st.error("선택한 시장의 종목 목록이 비어 있습니다.")
            st.stop()
        total_universe = len(universe)
        requested_count = total_universe if full_universe else min(int(sample_count), total_universe)
        count_label = "전체" if full_universe else str(requested_count)
        selected = universe.head(requested_count).copy().reset_index(drop=True)
        universe_fingerprint = hashlib.sha256("\n".join(universe["티커"].astype(str).tolist()).encode("utf-8")).hexdigest()
        config = {
            "버전": engine.SCRIPT_VERSION, "시장": market, "요청종목수": count_label,
            "목록전체종목수": total_universe, "목록출처": universe_source,
            "목표수익률_%": float(target_pct), "최대보유거래일": int(max_holding),
            "수수료_%_편도": float(fee_pct), "슬리피지_%_편도": float(slippage_pct),
            "ATR배수_v3_6_2": engine.ATR_MULTIPLE, "부트스트랩반복수": engine.BOOTSTRAP_REPS,
            "부트스트랩블록길이_거래일": int(max_holding), "유니버스_SHA256": universe_fingerprint,
            "v3_6_2_SHA256": engine.CORE_SOURCE_SHA256,
            "주가출처": "FinanceDataReader: fdr.DataReader(ticker)",
            "종목목록출처": universe_source, "현재구성종목기준": True,
        }
        output_dir = Path(__file__).parent / "screen_output" / market.replace(" ", "")
        checkpoint_path = output_dir / f"checkpoint_{market.replace(' ', '')}.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        records = empty_records(engine, checkpoint_path, config)
        allowed = set(selected["티커"].astype(str))
        for key in ["summary", "segments", "folds", "holdout", "thresholds", "trades", "errors"]:
            records[key] = [r for r in records.get(key, []) if str(r.get("티커", "")) in allowed]
        records["processed_tickers"] = [str(t) for t in records.get("processed_tickers", []) if str(t) in allowed]
        processed = set(records["processed_tickers"])
        todo = selected[~selected["티커"].astype(str).isin(processed)]

        st.markdown('<div class="section-title">분석 진행 상황</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("시장 목록 종목 수", f"{total_universe:,}")
        m2.metric("이번 실행 대상", f"{len(selected):,}")
        m3.metric("기존 결과 재사용", f"{len(processed):,}")
        if full_universe:
            st.warning(f"전체 {len(selected):,}개 종목 분석을 시작합니다. 종목별 모델 학습 때문에 매우 오래 걸릴 수 있습니다. 브라우저를 닫거나 서버가 재시작되면 이어서 실행해야 할 수 있습니다.")
        st.caption(f"종목 목록 출처: {universe_source}. 현재 구성종목 목록 기준이므로 과거 상장폐지·구성종목 변경에 따른 생존편향 가능성이 있습니다.")
        progress = st.progress(0.0, text="분석 준비 중")
        current_line = st.empty()
        status_box = st.empty()
        total_todo = len(todo)
        for idx, (_, stock_row) in enumerate(todo.iterrows(), start=1):
            meta = stock_row.to_dict()
            ticker = str(meta.get("티커", ""))
            company = str(meta.get("기업명", ticker))
            current_line.markdown(f"**진행 중:** {idx:,}/{total_todo:,} · `{ticker}` · {company}")
            try:
                summary, segments, folds, holdouts, trades, errors = engine.analyze_one(
                    core, meta, float(target_pct), int(max_holding), float(fee_pct), float(slippage_pct)
                )
                records["summary"].append(engine._safe(summary))
                records["segments"].extend(engine._safe(segments))
                records["folds"].extend(engine._safe(folds))
                records["holdout"].extend(engine._safe([r for r in holdouts if r.get("임계값선택목표") == "정밀도"]))
                records["thresholds"].extend(engine._safe(holdouts))
                records["trades"].extend(engine._safe(trades))
                records["errors"].extend(engine._safe(errors))
                detail_status = summary.get("분석상태", "상태 미확인")
                label_n = summary.get("전체유효라벨표본수", "-")
                holdout_n = summary.get("홀드아웃표본수", "-")
            except Exception as exc:
                records["summary"].append({"시장": market, "티커": ticker, "기업명": company,
                    "분석상태": "오류", "오류메시지": f"{type(exc).__name__}: {exc}"[:1500]})
                records["errors"].append({"시장": market, "티커": ticker, "기업명": company,
                    "분석상태": "오류", "세부메시지": f"{type(exc).__name__}: {exc}"[:1500],
                    "조치": "다음 실행에서 체크포인트 재사용 여부를 확인하고 해당 종목을 재검토하세요."})
                detail_status, label_n, holdout_n = "오류", "-", "-"
            records.setdefault("processed_tickers", []).append(ticker)
            processed.add(ticker)
            if idx % 5 == 0 or idx == total_todo:
                engine._checkpoint_save(checkpoint_path, config, records)
            progress.progress(idx / max(total_todo, 1), text=f"진행 {idx:,}/{total_todo:,}")
            status_box.caption(f"최근 처리: {detail_status} · 유효 라벨 {label_n} · 홀드아웃 표본 {holdout_n}")
        engine._checkpoint_save(checkpoint_path, config, records)

        for key, cols in [("summary", ["티커"]), ("segments", ["티커", "구간"]),
                          ("folds", ["티커", "차수"]), ("holdout", ["티커"]),
                          ("thresholds", ["티커", "임계값선택목표"]), ("trades", ["티커"]), ("errors", ["티커"])]:
            records[key] = dedupe_rows(records.get(key, []), cols)

        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"stock_screener_{market.replace(' ', '')}_{timestamp}.xlsx"
        with st.spinner("전체 순위와 Excel 보고서를 만들고 검증하고 있습니다..."):
            engine.build_workbook(output_path, records, config, engine.CORE_SOURCE_SHA256, universe_source)
            raw_summary = pd.DataFrame(records.get("summary", []))
            raw_folds = pd.DataFrame(records.get("folds", []))
            ranked = engine._screen_universe(raw_summary, raw_folds) if not raw_summary.empty else pd.DataFrame()
        st.session_state["scan_result"] = {
            "ranked": ranked,
            "records": records,
            "market": market,
            "universe_total": total_universe,
            "processed_count": len(raw_summary),
            "config": config,
            "excel_bytes": output_path.read_bytes(),
            "excel_name": output_path.name,
            "universe_source": universe_source,
        }
        st.success("분석과 Excel 보고서 생성이 완료되었습니다.")
    except Exception as exc:
        st.error(f"분석을 시작하거나 완료하는 중 오류가 발생했습니다: {type(exc).__name__}: {exc}")
        with st.expander("오류 상세"):
            st.code(traceback.format_exc())

result = st.session_state.get("scan_result")
if result:
    # Streamlit reruns the script on download-button clicks; restore the engine
    # in this render pass because run_scan is False on subsequent reruns.
    engine = get_engine()
    ranked = result["ranked"]
    st.markdown('<div class="section-title">분석 결과 요약</div>', unsafe_allow_html=True)
    if ranked is not None and not ranked.empty:
        group_col = "스크리닝그룹"
        groups = ranked[group_col].astype(str) if group_col in ranked.columns else pd.Series(dtype=str)
        success_col = ranked.get("분석상태", pd.Series("", index=ranked.index)).astype(str).eq("성공")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("분석 완료 종목", f"{len(ranked):,}")
        k2.metric("매수검토", f"{int((groups == '매수검토').sum()):,}")
        k3.metric("관찰", f"{int((groups == '관찰').sum()):,}")
        k4.metric("매수제외", f"{int((groups == '매수제외').sum()):,}")
        preferred = ["전체종합순위", "스크리닝그룹", "그룹별순위", "티커", "기업명", "홀드아웃정밀도_%",
                     "홀드아웃기저율_%", "평균순수익률_%", "CAGR_%", "MDD_%", "Calmar",
                     "현재모델점수_%", "최종임계값_%", "스크리닝판정사유"]
        show_cols = [c for c in preferred if c in ranked.columns]
        show_df = ranked[show_cols].copy() if show_cols else ranked.copy()
        show_df = engine._rename_display_columns(show_df)
        st.dataframe(show_df, width="stretch", hide_index=True, height=560)
        st.caption("그룹 순서는 매수검토 → 관찰 → 매수제외입니다. 종합순위는 그룹 우선순위를 반영하며, 모델 점수는 성공확률로 보정된 값이 아닙니다.")
        with st.expander("전체 결과와 분석 기준", expanded=False):
            st.dataframe(engine._rename_display_columns(ranked), width="stretch", hide_index=True)
            st.write("**분류 원칙**")
            st.dataframe(engine.build_screening_criteria_sheet(), width="stretch", hide_index=True)
        with st.expander("분석 실패·수집 오류 기록", expanded=False):
            errors_df = pd.DataFrame(result["records"].get("errors", []))
            if errors_df.empty:
                st.info("기록된 실패/수집 오류가 없습니다.")
            else:
                st.dataframe(errors_df, width="stretch", hide_index=True)
    else:
        st.warning("결과 요약을 만들 데이터가 없습니다. 분석 대상과 오류 기록을 확인하세요.")
    st.download_button("📊 전체 분석 Excel 다운로드", data=result["excel_bytes"],
        file_name=result["excel_name"], mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary", width="stretch")
    st.caption(f"시장: {result['market']} · 분석 기록: {result['processed_count']:,}개 · 종목 목록: {result['universe_source']}")
else:
    st.markdown("""
    <div class="glass-card">
      <div class="section-title">시작 방법</div>
      <p>1. 시장을 선택합니다: KOSPI 또는 S&amp;P 500.</p>
      <p>2. 먼저 30개 안팎으로 시험 실행해 데이터 연결과 결과표를 확인합니다.</p>
      <p>3. 이상이 없으면 <b>전체 구성종목 분석</b>을 선택해 다시 실행합니다.</p>
      <p class="small-note">전체 분석은 각 종목에 대해 모델 학습·홀드아웃·워크포워드 평가를 수행하므로 상당한 시간이 필요합니다. 체크포인트는 가능한 경우 이미 완료한 종목의 결과를 재사용합니다.</p>
    </div>
    """, unsafe_allow_html=True)
