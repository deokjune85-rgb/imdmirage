# ======================================================
# 🛡️ 베리타스 엔진 v9.5 — Operational Boot Restoration
# ======================================================
import streamlit as st
import google.generativeai as genai
import numpy as np

# ======================================================
# 1. 시스템 설정
# ======================================================
st.set_page_config(page_title="베리타스 엔진 9.5", page_icon="🛡️", layout="centered")

# 기본 스타일 (v7.0 감성 + 시각 통일)
st.markdown("""
<style>
#MainMenu, footer, header, .stDeployButton {visibility:hidden;}
html, body, div, span, p {
    font-family: 'Noto Sans KR', sans-serif !important;
    color: #FFFFFF !important;
    font-size: 17px !important;
    line-height: 1.6 !important;
}
[data-testid="stChatMessage"], [data-testid="stChatMessageContent"] {
    background-color: inherit !important;
    border: none !important;
}
h1 {
    text-align: left !important;
    font-weight: 900 !important;
    font-size: 34px !important;
    color: #FFFFFF !important;
    margin-top: 5px !important;
}
.lineblock {
    white-space: pre-wrap;
    opacity: 0;
    animation: fadeIn 0.6s forwards ease-in-out;
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
setInterval(scrollToBottom, 400);
</script>
""", unsafe_allow_html=True)

# ======================================================
# 2. 타이틀 및 경고
# ======================================================
st.title("베리타스 엔진 버전 9.5")
st.error("보안 경고: 본 시스템은 격리된 사설 환경(The Vault)에서 작동합니다. 모든 데이터는 기밀로 취급되며 외부로 유출되지 않습니다.")

# ======================================================
# 3. API 설정
# ======================================================
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("시스템 오류: 'GOOGLE_API_KEY' 누락. [Secrets] 탭 확인 필요.")
    st.stop()

genai.configure(api_key=API_KEY)

# ======================================================
# 4. 모델 초기화
# ======================================================
if "model" not in st.session_state:
    st.session_state.model = genai.GenerativeModel(
        "gemini-2.5-flash",
        system_instruction="당신은 법률 AI 시스템 '베리타스 엔진'입니다."
    )

if "chat" not in st.session_state:
    st.session_state.chat = st.session_state.model.start_chat(history=[])
    st.session_state.messages = []

    # ✅ 시스템 자동 부팅 메시지 (Phase 0)
    st.session_state.messages.append({
        "role": "Architect",
        "content": "시스템 초기화: 시뮬레이션 도메인 선택.\n\n분석을 진행할 사건의 법률/재무/의료 분야를 선택하십시오.\n\n1. 이혼 및 가사법 (Divorce/Family Law)\n2. 형사 변호 (Criminal Defense)\n3. 파산 및 회생 (Bankruptcy/Insolvency)\n4. 지적 재산권 (IP/Patent)\n5. 의료 소송 (Medical Malpractice)\n6. 세무 및 회계 (Tax/Accounting)\n7. 행정 소송 (Administrative Law)\n\n번호 또는 원하시는 분야를 입력하십시오."
    })

# ======================================================
# 5. 출력 루프
# ======================================================
for msg in st.session_state.messages:
    role = "Client" if msg["role"] == "user" else "Architect"
    avatar = "👤" if msg["role"] == "user" else "🛡️"
    with st.chat_message(role, avatar=avatar):
        st.markdown(f"<div class='lineblock'>{msg['content']}</div>", unsafe_allow_html=True)

# ======================================================
# 6. 입력 및 응답
# ======================================================
if prompt := st.chat_input("시뮬레이션 변수를 입력하십시오."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("Client", avatar="👤"):
        st.markdown(f"<div class='lineblock'>{prompt}</div>", unsafe_allow_html=True)

    with st.spinner("Architect 시스템 연산 중..."):
        try:
            stream = st.session_state.chat.send_message(prompt, stream=True)
            with st.chat_message("Architect", avatar="🛡️"):
                placeholder = st.empty()
                answer = ""
                for chunk in stream:
                    answer += chunk.text
                    placeholder.markdown(f"<div class='lineblock'>{answer}</div>", unsafe_allow_html=True)
                placeholder.markdown(f"<div class='lineblock'>{answer}</div>", unsafe_allow_html=True)
            st.session_state.messages.append({"role": "Architect", "content": answer})
        except Exception as e:
            st.error(f"시뮬레이션 오류: {e}")
