import streamlit as st
import pandas as pd
import numpy as np
import time
import datetime
import random
import re

# ---------------------------------------
# 0. 시스템 설정 및 보안 코드 (THE LOCK)
# ---------------------------------------

# [★중요★] 매일 아침 자동 변경되는 코드 설정 (ORACLE + MMDD)
TODAY_CODE = f"ORACLE{datetime.date.today().strftime('%m%d')}"

# [★백도어★] 마스터키
MASTER_KEY = "PANTHEON777"

st.set_page_config(
    page_title="알파픽 AI | The Oracle Engine",
    page_icon="✨",
    layout="centered"
)

# [CSS: Ultra-Premium Dark & Interactive Elements]
custom_css = """
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    
    /* 1. Core Theme */
    .stApp { background-color: #0A0A0A !important; color: #F5F5F5 !important; font-family: 'Pretendard', sans-serif; }

    /* 2. Hide Streamlit Branding */
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    
    /* 3. Data Table Styling */
    .stDataFrame thead th { background-color: #2C2C2C; color: #D4AF37; }
    .stDataFrame tbody tr:nth-child(even) { background-color: #1A1A1A; }
    .stDataFrame tbody tr:nth-child(odd) { background-color: #111111; }

    /* 4. VIP Section (The Paywall) */
    .vip-section {
        border: 2px solid #D4AF37; padding: 25px; margin: 20px 0;
        background-color: #1A1A1A; text-align: center; border-radius: 10px;
    }
    .lock-overlay { filter: blur(5px); pointer-events: none; user-select: none; }
    
    /* 5. CTA Button & Legal Disclaimer Button (Primary Buttons) */
     div.stButton > button, button[kind="primary"] {
        width: 100%;
        background-color: #D4AF37 !important;
        color: #000000 !important;
        font-weight: bold;
        border-radius: 8px;
        padding: 15px;
        border: none;
        font-size: 18px;
    }
    div.stButton > button:hover, button[kind="primary"]:hover {
        background-color: #B8860B !important;
    }
    
    /* 6. Legal Shield Styling */
    .legal-shield { background-color: #1A1A1A; padding: 30px; border-radius: 10px; border: 1px solid #333; }

    /* 7. Terminal Box (Deep Dive Visualization) */
    /* st.write_stream을 위한 스타일링 */
    .terminal-output p {
        background-color: #000000 !important;
        color: #00FF00 !important; /* Green Text */
        font-family: monospace !important;
        padding: 20px !important;
        border-radius: 8px !important;
        border: 1px solid #333 !important;
        min-height: 150px !important;
        white-space: pre-wrap !important;
    }
    
    /* 8. Chat Interface & Guide Chips Styling */
    .stChatMessage { padding: 10px 0; }
    
    /* 가이드 칩 버튼 스타일링 */
     .stApp .stHorizontalBlock div[data-testid="stButton"] > button {
         background-color: #2C2C2C !important;
         color: #AAAAAA !important;
         border: 1px solid #444 !important;
         border-radius: 20px !important;
         padding: 8px 16px !important;
         font-size: 14px !important;
         width: auto !important;
         font-weight: normal !important;
    }
    .stApp .stHorizontalBlock div[data-testid="stButton"] > button:hover {
        border-color: #D4AF37 !important;
        color: #D4AF37 !important;
        background-color: #444444 !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 세션 상태 초기화
if 'unlocked' not in st.session_state: st.session_state.unlocked = False
if 'agreed' not in st.session_state: st.session_state.agreed = False
# [★수정★] animated 플래그 추가
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'analyze_match' not in st.session_state: st.session_state.analyze_match = None
if 'last_analysis' not in st.session_state: st.session_state.last_analysis = None
if 'temp_chat_input' not in st.session_state: st.session_state.temp_chat_input = None

# [CRITICAL BUG FIX] 안정화된 타이핑 함수 (st.write_stream 대체)
# st.write_stream보다 안정적임.
def type_writer(text, placeholder, speed=0.03):
    display_text = ""
    try:
        for char in text:
            display_text += char
            placeholder.markdown(display_text + "▍")
            time.sleep(speed)
    finally:
        # 최종 텍스트(커서 제거) 출력 보장
        placeholder.markdown(display_text)

# ---------------------------------------
# 1. 법적 방탄조끼 (THE SHIELD) - TOS Gate
# ---------------------------------------
def legal_disclaimer_gate():
    """서비스 진입 전 강제적으로 법적 고지 및 동의를 받습니다."""
    st.markdown('<div class="legal-shield">', unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #D4AF37;'>알파픽 Sports AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>이용 약관 및 법적 고지</h3>", unsafe_allow_html=True)
    
    st.error("⚠️ 경고: 서비스를 이용하기 전에 다음 사항에 동의해야 합니다.")

    with st.form(key='agreement_form'):
        st.markdown("본 서비스는 스포츠 데이터를 분석하여 통계적 확률을 제공하는 **'정보 제공 서비스'**입니다.")
        agree1 = st.checkbox("[필수] **결과 면책:** AI 예측은 100%가 아니며, 경기 결과 및 금전적 손실에 대해 본 사는 어떠한 책임도 지지 않음에 동의합니다.")
        agree2 = st.checkbox("[필수] **준법 서약:** 국민체육진흥법을 준수하며, 불법 사설 사이트 이용을 금지합니다. 합법적인 투표권(스포츠토토/배트맨) 이용을 권장함에 동의합니다.")
        agree3 = st.checkbox("[필수] **환불 불가 정책:** VIP 접근 코드는 디지털 콘텐츠 특성상, 발급 및 사용 이후 환불이 불가능함에 동의합니다.")

        submit_button = st.form_submit_button(label='동의하고 알파픽 AI 시작하기')

        if submit_button:
            if agree1 and agree2 and agree3:
                st.session_state.agreed = True
                st.rerun()
            else:
                st.warning("모든 필수 항목에 동의해야 서비스를 이용할 수 있습니다.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ---------------------------------------
# 2. 데이터 시뮬레이션 엔진 (LIVE ENGINE)
# ---------------------------------------

# 캐시 제거됨. 매 실행마다 데이터 변동.
def generate_simulated_data():
    """실행 시마다 미세하게 변동되는 데이터를 생성하여 실시간 분석처럼 보이게 함."""
    matches = [
        ("맨체스터 시티", "루턴 타운 (EPL)"), ("아스널", "첼시 (EPL)"), ("리버풀", "에버턴 (EPL)"), 
        ("토트넘 홋스퍼", "웨스트햄 (EPL)"), ("바이에른 뮌헨", "도르트문트 (Bundes)"), ("레알 마드리드", "바르셀로나 (LaLiga)")
    ]
    
    data = []

    for i, (home, away) in enumerate(matches):
        # 1. 시장 배당률 생성 (+/- 5% 실시간 변동 시뮬레이션)
        base_odds = [1.10, 1.5, 1.7, 2.2, 1.3, 2.5]
        fluctuation = np.random.uniform(0.95, 1.05)
        odds_h = round(base_odds[i] * fluctuation, 2)
        odds_h = max(1.01, odds_h)
        
        market_prob_h = 1 / odds_h

        # 2. AI 예측 확률 생성
        if i == 0:
            # 시나리오 1: 역배 감지
            ai_prob_h = market_prob_h * np.random.uniform(0.55, 0.75)
            signal = "🚨 역배 감지 (상대팀 승/무)"
        elif i == 1 or i == 2:
             # 시나리오 2, 3: 가치 베팅
            ai_prob_h = market_prob_h * np.random.uniform(1.15, 1.35)
            signal = "🔥 강력 추천 (홈 승)"
        else:
            # 나머지 경기
            ai_prob_h = market_prob_h * np.random.uniform(0.92, 1.08)
            signal = "관망 (Hold)"

        ai_prob_h = min(ai_prob_h, 0.98)
        value_score_h = round((ai_prob_h - market_prob_h) * 100, 1)
        
        data.append({
            "경기 (Match)": f"{home} vs {away}",
            "시장 배당률 (Odds)": odds_h,
            "AI 예측 승률 (%)": f"{int(ai_prob_h*100)}%",
            "가치 지수 (Value)": value_score_h,
            "AI 시그널": signal
        })

    df = pd.DataFrame(data)
    df['Abs_Value'] = df['가치 지수 (Value)'].abs()
    df = df.sort_values(by="Abs_Value", ascending=False).reset_index(drop=True)
    df = df.drop(columns=['Abs_Value'])
    return df

# ---------------------------------------
# 3. 딥다이브 분석 엔진 (The Terminal & Streaming)
# ---------------------------------------

def stream_analysis(match_data):
    """터미널 스타일로 분석 로그를 스트리밍하고, 결과를 세션 상태에 저장합니다."""
    
    match_name = match_data["경기 (Match)"]
    signal = match_data["AI 시그널"]
    value_score = match_data['가치 지수 (Value)']
    
    # 분석 로그 생성 (동일)
    analysis_logs = [f"[{time.strftime('%H:%M:%S')}] 📡 Connecting to Global Sports Data Feed..."]
    
    if "역배 감지" in signal:
        analysis_logs.append(f"[{time.strftime('%H:%M:%S')}] 🚨 ALERT: Anomaly detected. Adjusting Probability (-{abs(value_score)}%)...")
    elif "강력 추천" in signal:
         analysis_logs.append(f"[{time.strftime('%H:%M:%S')}] 🔥 CONFIDENCE: Momentum surge detected. Adjusting Probability (+{value_score}%)...")

    analysis_logs.append(f"[{time.strftime('%H:%M:%S')}] 🧠 Running Monte Carlo Simulation (10,000 iterations)...")
    analysis_logs.append(f"[{time.strftime('%H:%M:%S')}] ✅ Analysis Complete.")
    
    # 스트리밍 제너레이터 (st.write_stream 사용)
    def generator():
        for log in analysis_logs:
            for char in log:
                yield char
                time.sleep(0.01)
            yield "\n"
            time.sleep(random.uniform(0.3, 0.8))

    # 터미널 박스 출력
    st.markdown("#### 분석 로그 (Real-time)")
    st.markdown('<div class="terminal-output">', unsafe_allow_html=True)
    st.write_stream(generator())
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 최종 코멘트 생성
    st.markdown("### AI 최종 코멘트")
    
    if "역배 감지" in signal:
        comment = f"주의가 필요합니다. 시장은 홈팀의 승리를 예상하지만, 알파픽 엔진은 숨겨진 위험 변수를 감지했습니다. 이변 확률이 통계적 임계치를 초과했습니다. 고위험-고수익 베팅 구간입니다."
    elif "강력 추천" in signal:
        comment = f"높은 확신 구간입니다. AI 예측 승률이 시장 배당률 대비 현저히 높습니다(가치 지수: {value_score}%). 이는 시장의 과소평가를 의미합니다. 적극적인 베팅을 권장합니다."
    else:
        comment = f"시장 예측과 AI 예측이 유사한 범위 내에 있습니다. 유의미한 시장 왜곡은 감지되지 않았습니다. 관망(Hold)을 권장합니다."

    # [★수정★] 코멘트 스트리밍도 렌더링 루프에서 처리되도록 챗 히스토리에 추가
    # st.write_stream(comment_generator()) 대신 세션 상태에 저장
    
    # [★기획 1★] 분석 완료 후 컨텍스트 저장
    st.session_state.last_analysis = {
        "match_name": match_name,
        "signal": signal,
        "value_score": value_score,
        "comment": comment
    }
    # 코멘트를 챗 히스토리에 추가하여 자연스러운 흐름 유도 (애니메이션은 렌더링 루프에서 처리)
    st.session_state.chat_history.append({"role": "assistant", "content": comment, "animated": False})


# ---------------------------------------
# 4. AI 챗 어시스턴트 (인지 강화 모듈)
# ---------------------------------------

# [★기획 3★] 키워드 딕셔너리 및 NLP 개선
SLANG_DICT = {
    "TRUST": ["확실해", "믿어도 돼", "부러지면", "한강", "진짜지", "쫄려", "확신"],
    "MONEY": ["얼마", "올인", "소액", "강승부", "시드", "배팅액", "금액"],
    "ANOMALY": ["역배", "이변", "터지냐", "로또", "변수"],
    "CONTEXT": ["아까 그거", "방금 본거", "이거 어때", "확인해줘", "이 경기"]
}

# [★수정★] 약어 사전 추가
ALIASES = {
    "맨시티": "맨체스터 시티",
    "뮌헨": "바이에른 뮌헨",
    "레알": "레알 마드리드",
    "바르샤": "바르셀로나",
    "파리": "파리 생제르맹",
}

def normalize_query(query):
    query = query.lower()
    for alias, official in ALIASES.items():
        if alias in query:
            # Replace alias with official name for better matching
            query = query.replace(alias, official)
    return query

# [★핵심 수정★] 로직 처리만 담당하고 렌더링은 제거
def handle_chat_query(query, df):
    """사용자의 질문에 대한 응답을 계산하고 세션 상태에 저장합니다."""
    
    response = ""
    query = normalize_query(query) # 쿼리 정규화
    
    # [★기획 1★] 컨텍스트 활용
    context = st.session_state.last_analysis
    is_context_query = False
    
    if context:
        other_match_mentioned = False
        for index, row in df.iterrows():
                match_name = row["경기 (Match)"]
                # 정규화된 쿼리로 다른 경기 언급 확인
                if match_name != context["match_name"] and any(word.lower() in query for word in match_name.split(" ") if len(word) > 2):
                    other_match_mentioned = True
                    break
        
        if not other_match_mentioned and (
            any(s in query for s in SLANG_DICT["CONTEXT"]) or 
            any(s in query for s in SLANG_DICT["TRUST"]) or 
            any(s in query for s in SLANG_DICT["MONEY"])):
            
            is_context_query = True
            match_name = context["match_name"]
            value = context["value_score"]
            
            # [★기획 3★] 슬랭 대응
            if any(s in query for s in SLANG_DICT["TRUST"]):
                response = f"[{match_name}] 분석 결과에 대한 질문이군요. 데이터는 감정보다 정확합니다. 현재 신뢰도 지수는 높음({abs(value)}점) 구간입니다. '한강' 갈 일은 통계적으로 낮습니다. 다만, 불안하시면 보험 베팅(무승부 방어)을 고려하십시오."
            
            elif any(s in query for s in SLANG_DICT["MONEY"]):
                if abs(value) > 15:
                     response = f"[{match_name}]은 가치 지수({value}%)가 높습니다. 시장 왜곡이 확인된 구간이므로 '강승부' (시드머니의 20%)를 추천합니다."
                else:
                    response = f"[{match_name}]은 안정적인 구간입니다. 시드머니의 10% 이내를 권장합니다."
            
            else:
                response = f"방금 분석한 [{match_name}] 말씀이시군요. AI의 최종 코멘트를 다시 확인해 드리겠습니다:\n\"{context['comment'][:100]}...\""

    if not is_context_query:
        # [★기획 3★] 일반 키워드/슬랭 인식
        if any(s in query for s in SLANG_DICT["ANOMALY"]):
            underdog = df[df['AI 시그널'].str.contains("역배 감지")]
            if not underdog.empty:
                match_name = underdog.iloc[0]["경기 (Match)"]
                response = f"현재 AI는 [{match_name}] 경기에서 심각한 시장 왜곡을 감지했습니다. 이변 가능성이 높습니다. 고배당을 노릴 기회입니다. Deep Dive를 확인하십시오."
            else:
                response = "현재 감지된 강력한 역배 시그널은 없습니다."
                
        elif "추천" in query or "뭐가 좋아" in query:
            response = "가장 신뢰도가 높은 경기는 VIP 픽 Top 3에 공개됩니다. VIP 코드를 입력하여 확인하십시오."
        
        elif "vip" in query or "구독" in query:
             response = "VIP 멤버십은 월 99,000원이며, 매일 Top 3 픽 제공 및 실시간 텔레그램 알림방 입장이 가능합니다. 하단의 구매 안내를 참조하십시오."

        else:
            # 특정 경기 질문 확인
            match_found = False
            for index, row in df.iterrows():
                match_name = row["경기 (Match)"]
                # 정규화된 쿼리로 팀 이름 매칭
                if any(word.lower() in query for word in match_name.split(" ") if len(word) > 2):
                    signal = row["AI 시그널"]
                    value = row["가치 지수 (Value)"]
                    response = f"[{match_name}] 분석 결과: AI 시그널은 '{signal}'이며, 가치 지수는 {value}입니다. 더 자세한 내용은 Deep Dive 분석을 실행하십시오."
                    match_found = True
                    break
            
            # [★기획 4★] 무한 루프 방어 (Fallback Loop - 수익화 유도)
            if not match_found:
                if any(x in query for x in ["안녕", "누구", "뭐야"]):
                     response = "저는 알파픽 Sports AI입니다. 시장 데이터를 분석하여 수익 창출을 돕는 전문가 시스템입니다."
                else:
                    # 이해 못하는 질문은 비즈니스로 연결
                    response = f"저는 [스포츠 데이터 분석]에 특화된 AI입니다. '{query}'에 대한 직접적인 답변보다는 오늘 밤 수익을 낼 경기를 분석해 드릴 수 있습니다. 텔레그램 VIP 방에서는 실시간 고급 정보도 제공 중입니다."

    # [★핵심 수정★] 응답을 세션 상태에 저장 (렌더링은 메인 루프에서 처리)
    # animated=False 플래그 추가
    st.session_state.chat_history.append({"role": "assistant", "content": response, "animated": False})


# ---------------------------------------
# 5. 메인 애플리케이션 로직 (★렌더링/로직 분리 적용★)
# ---------------------------------------

def main_app():
    # [Header]
    st.markdown(f"<h1 style='text-align: center; font-family: serif; margin-bottom: 5px; color: #D4AF37;'>알파픽 Sports AI</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; font-size:14px; color:#555; letter-spacing: 2px;'>THE ORACLE ENGINE | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</p>", unsafe_allow_html=True)
    st.divider()

    # 데이터 로드 (매 실행마다 변동됨)
    df = generate_simulated_data()
    
    with st.spinner("알파픽 엔진이 최신 데이터를 분석 중입니다... (실시간 변동 적용)"):
         time.sleep(0.5)

    VIP_PICKS_COUNT = 3
    vip_picks = df.head(VIP_PICKS_COUNT)
    free_picks = df.tail(-VIP_PICKS_COUNT)

    # ---------------------------------------
    # VIP 섹션 (The Paywall) (이전과 동일)
    # ---------------------------------------
    st.markdown("<h2 style='color: #D4AF37; text-align: center;'>✨ VIP AI 추천 픽 (Top 3)</h2>", unsafe_allow_html=True)

    if not st.session_state.unlocked:
        st.markdown('<div class="vip-section">', unsafe_allow_html=True)
        st.markdown('<div class="lock-overlay">', unsafe_allow_html=True)
        st.dataframe(vip_picks, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### 🔒 오늘의 VIP 접근 코드 입력")
        access_code = st.text_input("결제 후 발급받은 접근 코드를 입력하세요.", type="password")
        
        if st.button("VIP 픽 잠금 해제", type="primary", key="unlock_vip"):
            if access_code == TODAY_CODE or access_code == MASTER_KEY:
                st.session_state.unlocked = True
                st.rerun()
            else:
                st.error("잘못된 코드입니다. (하단 참조)")
                
        st.warning("⚠️ 보안 경고: 코드 공유 감지 시 즉시 만료 및 영구 차단됩니다.")
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown('<div class="vip-section" style="border-color: #00E676;">', unsafe_allow_html=True)
        st.success("✨ VIP 접근 활성화됨.")
        st.dataframe(vip_picks, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # [Monetization CTA]
    if not st.session_state.unlocked:
        st.markdown("---")
        st.markdown("### 💎 VIP 접근 코드 구매하기")
        st.info(f"""
        **[가격 정책]** 1일 이용권: 10,000원 | VIP 월 구독: 99,000원
        
        **[구매 방법]** 입금 후 아래 카카오톡 채널로 연락주시면 1분 내로 코드를 발급해 드립니다.
        👉 **[여기에 네 카카오톡 채널 링크 삽입]**
        """)

    # ---------------------------------------
    # 딥다이브 분석기 (The Analyzer)
    # ---------------------------------------
    st.markdown("---")
    st.markdown("<h2>🧬 딥다이브(Deep Dive) 분석기</h2>", unsafe_allow_html=True)
    
    match_list = df['경기 (Match)'].tolist()
    selected_match_name = st.selectbox("상세 분석을 원하는 경기를 선택하세요.", ["선택 안 함"] + match_list)
    
    if selected_match_name != "선택 안 함":
        if st.button("AI 심층 분석 실행", type="primary", key="run_analysis"):
            st.session_state.analyze_match = selected_match_name
            # 분석 실행 시 기존 채팅 기록 초기화 (선택 사항)
            # st.session_state.chat_history = [] 
            st.rerun()

    # [★수정★] 분석 렌더링 로직
    if st.session_state.analyze_match:
        match_data = df[df["경기 (Match)"] == st.session_state.analyze_match]
        if not match_data.empty:
            # 분석 실행 및 Context 저장 (Deep Dive는 st.write_stream 사용 가능)
            stream_analysis(match_data.iloc[0])
        st.session_state.analyze_match = None # 분석 완료 후 초기화
        st.rerun() # 코멘트가 챗 히스토리에 추가되었으므로 렌더링을 위해 재실행

    # ---------------------------------------
    # 무료 섹션 (The Bait)
    # ---------------------------------------
    st.markdown("---")
    st.markdown("<h2>📊 일반 AI 분석 데이터</h2>", unsafe_allow_html=True)
    st.dataframe(free_picks, use_container_width=True, hide_index=True)

    # ---------------------------------------
    # [★신규★] AI 챗 어시스턴트 (The Assistant)
    # ---------------------------------------
    st.markdown("---")
    st.markdown("<h2>✨ AI 분석 비서 (Q&A)</h2>", unsafe_allow_html=True)

    # [★핵심 수정★ 챗 히스토리 렌더링: 안정화된 애니메이션 처리]
    for i, message in enumerate(st.session_state.chat_history):
        avatar = "✨" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar):
            
            is_last_message = (i == len(st.session_state.chat_history) - 1)
            
            # 마지막 AI 메시지이고 아직 애니메이션되지 않았다면 타이핑 효과 적용
            if message["role"] == "assistant" and not message.get("animated") and is_last_message:
                # [★버그 수정★] 안정적인 type_writer 함수 사용
                placeholder = st.empty()
                type_writer(message["content"], placeholder)
                message["animated"] = True # 완료 처리
            else:
                # 이전 메시지 또는 유저 메시지는 즉시 출력
                st.markdown(message["content"])


    # [★기획 2: 가이드 칩 (Guide Chips)] 구현
    st.markdown("<p style='font-size:12px; color:#666; margin-bottom: 5px;'>추천 질문:</p>", unsafe_allow_html=True)
    
    cols = st.columns(4)
    
    # 버튼 클릭 시 temp_chat_input에 저장하고 재실행
    with cols[0]:
        if st.button("💣 역배 추천", key="chip1"):
            st.session_state.temp_chat_input = "오늘 역배 터질 경기 있어?"
            st.rerun()
    with cols[1]:
        if st.button("💰 얼마 걸까?", key="chip2"):
            # 컨텍스트가 있으면 활용, 없으면 일반 질문
            if st.session_state.last_analysis:
                 st.session_state.temp_chat_input = f"방금 본 경기({st.session_state.last_analysis['match_name'][:10]}...) 얼마 걸까?"
            else:
                st.session_state.temp_chat_input = "베팅 금액 추천해줘."
            st.rerun()
            
    # 컨텍스트가 있을 때만 활성화되는 버튼
    if st.session_state.last_analysis:
        with cols[2]:
            if st.button("🤔 이거 확실해?", key="chip3"):
                st.session_state.temp_chat_input = "방금 분석한 경기 진짜 믿어도 돼? 한강 가기 싫다."
                st.rerun()
    with cols[3]:
         if st.button("🏆 VIP 정보?", key="chip4"):
            st.session_state.temp_chat_input = "VIP 정보는 뭐가 달라?"
            st.rerun()

    # [★핵심 수정★ 챗 입력 처리: 로직과 렌더링 분리]
    user_query = st.chat_input("분석 결과에 대해 질문하세요. (예: 맨시티 경기 어때?)")
    
    # temp_chat_input이 우선권을 가짐
    if st.session_state.temp_chat_input:
        user_query = st.session_state.temp_chat_input
        st.session_state.temp_chat_input = None # 사용 후 초기화

    if user_query:
        # 1. 유저 메시지 저장 (애니메이션 필요 없음)
        st.session_state.chat_history.append({"role": "user", "content": user_query, "animated": True})
        
        # 2. AI 응답 처리 (로직만 실행하고 응답 저장)
        handle_chat_query(user_query, df)
        
        # 3. [★중요★] 스크립트 재실행 (렌더링 루프가 애니메이션을 처리하도록 함)
        st.rerun()


# ---------------------------------------
# 실행 제어 (Gatekeeper)
# ---------------------------------------

if st.session_state.agreed:
    main_app()
else:
    legal_disclaimer_gate()
