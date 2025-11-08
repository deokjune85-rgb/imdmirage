# ======================================================
# 🛡️ 베리타스 엔진 v10.0 — Phase Protocol Reinforced Build
# ======================================================
import streamlit as st
import time

# ======================================================
# 1. SYSTEM INIT
# ======================================================
st.set_page_config(page_title="베리타스 엔진 10.0", page_icon="🛡️", layout="centered")

# CSS 통일
st.markdown("""
<style>
#MainMenu, footer, header, .stDeployButton {visibility:hidden;}
html, body, div, span, p {
    font-family: 'Noto Sans KR', sans-serif !important;
    color: #FFFFFF !important;
    font-size: 17px !important;
    line-height: 1.7 !important;
}
h1 {
    text-align: left !important;
    font-weight: 900 !important;
    font-size: 34px !important;
    margin-top: 10px !important;
    margin-bottom: 15px !important;
    color: #FFFFFF !important;
}
.lineblock {
    white-space: pre-wrap;
    margin-bottom: 5px;
    opacity: 0;
    animation: fadeIn 0.5s forwards ease-in-out;
}
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
</style>
""", unsafe_allow_html=True)

# 자동 스크롤
st.markdown("""
<script>
const scrollToBottom = () => {
  var chat = window.parent.document.querySelector('[data-testid="stVerticalBlock"]');
  if (chat) chat.scrollTo(0, chat.scrollHeight);
};
setInterval(scrollToBottom, 300);
</script>
""", unsafe_allow_html=True)

# ======================================================
# 2. UI TITLE
# ======================================================
st.title("베리타스 엔진 버전 10.0")
st.error("보안 경고: 본 시스템은 격리된 사설 환경(The Vault)에서 작동합니다. 모든 데이터는 기밀로 취급되며 외부로 유출되지 않습니다.")

# ======================================================
# 3. PHASE CONTROL
# ======================================================
if "phase" not in st.session_state:
    st.session_state.phase = "0"

def show_phase_0():
    st.markdown("""
**시스템 초기화: 시뮬레이션 도메인 선택.**

분석을 진행할 사건의 법률/재무/의료 분야를 선택하십시오.

1. 이혼 및 가사법 (Divorce/Family Law)  
2. 형사 변호 (Criminal Defense)  
3. 파산 및 회생 (Bankruptcy/Insolvency)  
4. 지적 재산권 (IP/Patent)  
5. 의료 소송 (Medical Malpractice)  
6. 세무 및 회계 (Tax/Accounting)  
7. 행정 소송 (Administrative Law)

번호 또는 원하시는 분야를 입력하십시오.
""")

def show_phase_05():
    st.markdown("""
**Phase 0.5: 형사 세부 분야 선택.**

2-1. 마약 (투약/소지/매매/알선)  
2-2. 성범죄 및 스토킹  
2-3. 음주운전  
2-4. 도박 (사이버/오프라인)  
2-5. 금융/경제 범죄 (자본시장법, 사기/횡령/배임, 특금법)  
2-6. 명예훼손 및 정보통신망법 위반  
2-7. 유사수신  
2-8. 기타 일반 형사 (폭행 등)
""")

def show_phase_1():
    st.markdown("""
**Phase 1: 핵심 변수 입력.**

1/6. 현재 문제가 된 '혐의 내용'은 무엇입니까?  
(예: 유사수신행위법 위반 및 특경법 사기)
""")

# ======================================================
# 4. PHASE FLOW
# ======================================================
if st.session_state.phase == "0":
    show_phase_0()

elif st.session_state.phase == "0.5":
    show_phase_05()

elif st.session_state.phase == "1":
    show_phase_1()

# ======================================================
# 5. USER INPUT (STRICT CONTROL)
# ======================================================
if user_input := st.chat_input("시뮬레이션 변수를 입력하십시오."):
    if st.session_state.phase == "0":
        if user_input.strip() == "2":
            st.session_state.phase = "0.5"
            st.rerun()
        else:
            st.warning("올바른 도메인 번호를 입력하십시오. (예: 2)")
    elif st.session_state.phase == "0.5":
        st.session_state.phase = "1"
        st.rerun()
    elif st.session_state.phase == "1":
        st.success("Phase 1 입력 완료. 다음 단계로 진행 중...")
        time.sleep(1)
        st.session_state.phase = "2"
        st.rerun()
