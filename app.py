import streamlit as st
import google.generativeai as genai
import os

# --- 1. 시스템 설정 (The Vault & Mirage Protocol) ---
# 페이지 타이틀과 레이아웃 설정
st.set_page_config(page_title="아이엠디 아키텍처 버전 7.0", page_icon="🛡️", layout="centered")

# CSS 해킹: Streamlit 기본 로고, 메뉴, 헤더, 푸터를 완벽하게 숨긴다. (신기루 프로토콜)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stDeployButton {visibility: hidden;} /* 배포 버튼 숨기기 */
            /* 필요시 여기에 추가적인 커스텀 스타일링(예: 다크 모드)을 적용하라. */
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- 2. 타이틀 및 경고 (황제의 교리) ---
st.title("아이엠디 아키텍처 버전 7.0")
# st.error를 사용하여 강력한 시각적 경고 표시
st.error("보안 경고: 본 시스템은 격리된 사설 환경(The Vault)에서 작동합니다. 모든 데이터는 기밀로 취급되며 외부로 유출되지 않습니다.")

# --- 3. API 키 및 모델 설정 (The Engine & EPE/KB) ---
# Streamlit Secrets를 사용하여 API 키를 안전하게 로드한다. (코드에 직접 키를 넣지 않음)
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("시스템 오류: 엔진 연결 실패. (API Key 누락)")
    st.stop()

genai.configure(api_key=API_KEY)

# 외부 파일 로드 (프라임 게놈 전체를 system_prompt.txt에 저장)
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_INSTRUCTION = f.read()

# 모델 초기화
if "model" not in st.session_state:
    st.session_state.model = genai.GenerativeModel('models/gemini-1.5-flash-latest',
                                                   system_instruction=SYSTEM_INSTRUCTION)

# --- 4. 대화 세션 관리 및 자동 시작 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat" not in st.session_state:
    st.session_state.chat = st.session_state.model.start_chat(history=[])
    
    # 시스템 초기 메시지(Phase 0)를 강제로 생성하여 시작한다. (자동 시작 프로토콜)
    # 초기화 메시지를 보냄 (사용자에게는 보이지 않음)
    initial_prompt = "시스템 가동. '동적 라우팅 프로토콜'을 실행하여 Phase 0를 시작하라."
    try:
        response = st.session_state.chat.send_message(initial_prompt)
        # 첫 응답(Phase 0 안내)을 기록에 추가한다.
        st.session_state.messages.append({"role": "Architect", "content": response.text})
    except Exception as e:
        st.error(f"시스템 초기화 실패: {e}")

# 이전 대화 기록 표시 (역할 이름을 커스텀)
for message in st.session_state.messages:
    role_name = message["role"]
    avatar = "🛡️" # Architect 아바타
    if role_name == "user":
        role_name = "Client"
        avatar = "👤" # Client 아바타
        
    with st.chat_message(role_name, avatar=avatar):
        st.markdown(message["content"])

# --- 5. 사용자 입력 및 응답 생성 (스트리밍 적용) ---
if prompt := st.chat_input("시뮬레이션 변수를 입력하십시오."):
    # 사용자 입력 표시 및 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("Client", avatar="👤"):
        st.markdown(prompt)

    # 시스템 응답 생성 (API 호출)
    with st.spinner("Architect 시스템 연산 중... 변수 분석 및 시뮬레이션 실행..."):
        try:
            # 스트리밍 사용(stream=True)으로 응답 속도 개선
            response_stream = st.session_state.chat.send_message(prompt, stream=True)
            
            # 시스템 응답 표시 및 저장
            with st.chat_message("Architect", avatar="🛡️"):
                # 스트리밍 응답을 위한 플레이스홀더 생성
                response_placeholder = st.empty()
                full_response = ""
                # 스트림을 순회하며 실시간으로 화면에 출력
                for chunk in response_stream:
                    full_response += chunk.text
                    # 타이핑 효과처럼 보이게 함
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "Architect", "content": full_response})
        
        except Exception as e:
            error_msg = f"시뮬레이션 오류 발생. 시스템 로그 확인 필요: {e}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "Architect", "content": error_msg})
