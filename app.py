# ======================================================
# 🛡️ 베리타스 엔진 v8.7 — Final Stable Build
# ======================================================
import streamlit as st
import google.generativeai as genai
import requests, numpy as np

# ======================================================
# 1. SYSTEM CONFIG
# ======================================================
st.set_page_config(page_title="베리타스 엔진", page_icon="🛡️", layout="centered")

# ✅ 전체 스타일 통합
st.markdown("""
<style>
#MainMenu, footer, header, .stDeployButton {visibility:hidden;}

html, body, div, span, p {
    font-family: 'Noto Sans KR', sans-serif !important;
    font-size: 16px !important;
    line-height: 1.6 !important;
    color: #FFFFFF !important;
}

[data-testid="stChatMessage"], [data-testid="stChatMessageContent"] {
    background-color: inherit !important;
    border: none !important;
}

/* ✅ 자연스러운 텍스트 표시 (Fade-in 효과) */
.lineblock {
    white-space: pre-wrap;
    line-height: 1.6;
    margin-bottom: 4px;
    color: #FFFFFF;
    font-size: 16px;
    opacity: 0;
    animation: fadeIn 0.6s forwards ease-in-out;
}
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}

/* ✅ 리스트 줄간격 완전 통일 */
.option-list {
    line-height: 1.6 !important;
    margin-top: 10px !important;
}
.option-list div {
    margin-bottom: 2px !important;
}

/* ✅ 메인 타이틀 */
.main-title {
    font-size: 26px !important;
    font-weight: 800 !important;
    color: #FFFFFF !important;
    text-align: center !important;
    margin-top: 15px !important;
    margin-bottom: 15px !important;
}

/* ✅ 자동 스크롤 보조 (채팅 갱신 시) */
.stChatMessage {
    scroll-margin-bottom: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ✅ 자동 스크롤 JS (맨 하단 자동 이동)
st.markdown("""
<script>
const scrollToBottom = () => {
  var chatContainer = window.parent.document.querySelector('[data-testid="stChatInput"]');
  if (chatContainer) {
    chatContainer.scrollIntoView({ behavior: "smooth", block: "end" });
  }
};
setInterval(scrollToBottom, 500);
</script>
""", unsafe_allow_html=True)

# ✅ 메인 타이틀 표시
st.markdown("<div class='main-title'>🛡️ 베리타스 엔진</div>", unsafe_allow_html=True)
st.caption("AI 법률 시뮬레이션 시스템 — Confidential Mode")

# ======================================================
# 2. API CONFIG
# ======================================================
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("시스템 오류: 'GOOGLE_API_KEY' 누락. [Secrets] 확인 필요.")
    st.stop()

genai.configure(api_key=API_KEY)

# ======================================================
# 3. MODEL INIT
# ======================================================
if "model" not in st.session_state:
    st.session_state.model = genai.GenerativeModel(
        "gemini-2.5-flash",
        system_instruction="당신은 법률 AI 시스템 '베리타스 엔진'입니다."
    )

if "chat" not in st.session_state:
    st.session_state.chat = st.session_state.model.start_chat(history=[])
    st.session_state.messages = []

# ======================================================
# 4. UI — 선택 섹션 예시
# ======================================================
with st.chat_message("Architect", avatar="🛡️"):
    st.markdown("""
    <div class='option-list'>
    <div>1. 이혼 및 가사법 (Divorce/Family Law)</div>
    <div>2. 형사 변호 (Criminal Defense)</div>
    <div>3. 파산 및 회생 (Bankruptcy/Insolvency)</div>
    <div>4. 지적 재산권 (IP/Patent)</div>
    <div>5. 의료 소송 (Medical Malpractice)</div>
    <div>6. 세무 및 회계 (Tax/Accounting)</div>
    <div>7. 행정 소송 (Administrative Law)</div>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# 5. CHAT LOOP
# ======================================================
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🛡️"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(f"<div class='lineblock'>{msg['content']}</div>", unsafe_allow_html=True)

if prompt := st.chat_input("시뮬레이션 변수를 입력하십시오."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(f"<div class='lineblock'>{prompt}</div>", unsafe_allow_html=True)

    with st.spinner("시스템 연산 중..."):
        try:
            response_stream = st.session_state.chat.send_message(prompt, stream=True)
            with st.chat_message("Architect", avatar="🛡️"):
                placeholder = st.empty()
                full_text = ""
                for chunk in response_stream:
                    full_text += chunk.text
                placeholder.markdown(f"<div class='lineblock'>{full_text}</div>", unsafe_allow_html=True)
            st.session_state.messages.append({"role": "Architect", "content": full_text})
        except Exception as e:
            st.error(f"시뮬레이션 오류: {e}")
