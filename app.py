# ======================================================
# 🛡️ 베리타스 엔진 v8.8 — 윤진 커스텀 버전
# ======================================================
import streamlit as st
import google.generativeai as genai
import requests, numpy as np

# ======================================================
# 1. SYSTEM CONFIG
# ======================================================
st.set_page_config(page_title="베리타스 엔진", page_icon="🛡️", layout="centered")

# ✅ 스타일 완전 커스텀
st.markdown("""
<style>
#MainMenu, footer, header, .stDeployButton {visibility:hidden;}

/* 전체 글꼴 및 색상 */
html, body, div, span, p {
    font-family: 'Noto Sans KR', sans-serif !important;
    color: #FFFFFF !important;
    line-height: 1.6 !important;
    font-size: 17px !important;
}

/* 메인 타이틀 — 왼쪽 정렬, 크고 두꺼움 */
.main-title {
    font-size: 32px !important;
    font-weight: 900 !important;
    color: #FFFFFF !important;
    text-align: left !important;
    margin-top: 10px !important;
    margin-bottom: 25px !important;
}

/* 채팅 메시지 스타일 */
[data-testid="stChatMessage"], [data-testid="stChatMessageContent"] {
    background-color: inherit !important;
    border: none !important;
}

/* 줄간격 통일 */
.option-list div {
    margin-bottom: 4px !important;
    line-height: 1.6 !important;
}

/* 텍스트 Fade-in */
.lineblock {
    white-space: pre-wrap;
    margin-bottom: 5px;
    opacity: 0;
    animation: fadeIn 0.7s forwards ease-in-out;
}
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}

/* 자동 스크롤 */
.stChatMessage {
    scroll-margin-bottom: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ✅ 자동 스크롤 유지 (채팅 입력 시)
st.markdown("""
<script>
const scrollToBottom = () => {
  var chatContainer = window.parent.document.querySelector('[data-testid="stVerticalBlock"]');
  if (chatContainer) chatContainer.scrollTo(0, chatContainer.scrollHeight);
};
setInterval(scrollToBottom, 400);
</script>
""", unsafe_allow_html=True)

# ✅ 메인 타이틀 (왼쪽 정렬)
st.markdown("<div class='main-title'>🛡️ 베리타스 엔진</div>", unsafe_allow_html=True)

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
# 4. LIST 출력 (1~7 줄바꿈 정상)
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

if prompt := st.chat_input(" "):  # 안내문 제거
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
