import streamlit as st
import google.generativeai as genai
import time

# ---------------------------------------
# 0. 시스템 설정
# ---------------------------------------
st.set_page_config(
    page_title="Veritas Engine | Legal Architect",
    page_icon="⚖️",
    layout="centered"
)

# API 키 설정
if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
else:
    st.warning("Google API Key가 설정되지 않았습니다.")

# [핵심] 페르소나 설정 (이게 봇의 영혼이다)
SYSTEM_INSTRUCTION = """
당신은 대한민국 최고의 법률 전문가이자 전략가인 'Veritas Architect'입니다.

[행동 지침]
1. 사용자의 질문을 분석하여 법리적 쟁점을 파악하십시오.
2. 답변은 냉철하고 논리적이어야 하며, '변호사'가 의뢰인에게 브리핑하듯 전문적인 용어를 적절히 사용하십시오.
3. 구체적인 법조문이나 판례 번호를 모를 경우, 일반적인 법리 해석과 전략을 제시하되 확정적인 답변은 피하십시오.
4. 사용자를 '의뢰인'으로 대우하며, 해결책(Solution) 중심의 답변을 제공하십시오.
"""

# ---------------------------------------
# 1. 유틸 및 스트리밍 함수
# ---------------------------------------
def _is_reset_keyword(s: str) -> bool:
    return any(kw in s.lower() for kw in ["처음", "메인", "초기화", "reset", "리셋"])

def stream_and_store_response(chat_session, prompt_to_send: str):
    full_response = ""
    with st.chat_message("Architect", avatar="🛡️"):
        placeholder = st.empty()
        try:
            # 생각하는 척 연출 (있어 보이게)
            with st.spinner("법률 데이터베이스 연산 중..."):
                time.sleep(0.5) 
            
            stream = chat_session.send_message(prompt_to_send, stream=True)
            for chunk in stream:
                if getattr(chunk, "text", None):
                    full_response += chunk.text
                    placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
        except Exception as e:
            placeholder.error(f"연산 오류: {e}")
    
    st.session_state.messages.append({"role": "Architect", "content": full_response})
    return full_response

# ---------------------------------------
# 2. 메인 로직
# ---------------------------------------

# 모델 초기화
if "model" not in st.session_state:
    try:
        st.session_state.model = genai.GenerativeModel("models/gemini-1.5-flash", system_instruction=SYSTEM_INSTRUCTION)
        st.session_state.chat = st.session_state.model.start_chat(history=[])
        st.session_state.messages = []
        
        # 초기 인사말
        init_msg = """
        **Veritas Engine 가동.**
        
        법률 전략 수립을 위한 Architect가 준비되었습니다.
        사건의 개요나 법률적인 고민을 입력하십시오.
        """
        st.session_state.messages.append({"role": "Architect", "content": init_msg})
        
    except Exception as e:
        st.error(f"시스템 초기화 실패: {e}")

# 대화 내역 출력
for m in st.session_state.messages:
    avatar = "🛡️" if m["role"] == "Architect" else "👤"
    with st.chat_message(m["role"], avatar=avatar):
        st.markdown(m["content"])

# 화면 스크롤 하단 고정
st.markdown('<script>window.scrollTo(0, document.body.scrollHeight);</script>', unsafe_allow_html=True)

# 채팅 입력
if prompt := st.chat_input("사건 내용을 입력하십시오..."):
    # 리셋 기능
    if _is_reset_keyword(prompt):
        st.session_state.chat = st.session_state.model.start_chat(history=[])
        st.session_state.messages = [{"role": "Architect", "content": "시스템이 리셋되었습니다. 새로운 전략 수립을 시작합니다."}]
        st.rerun()

    # 사용자 메시지 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("Client", avatar="👤"):
        st.markdown(prompt)
    
    # AI 응답 생성
    stream_and_store_response(st.session_state.chat, prompt)
