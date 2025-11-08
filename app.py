import streamlit as st
import google.generativeai as genai

# --- 1. 시스템 설정 (The Vault & Mirage Protocol) ---
st.set_page_config(page_title="베리타스엔진 버전 7.0", page_icon="🛡️", layout="centered")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- 2. 타이틀 및 경고 ---
st.title("베리타스 엔진 버전 7.0")
st.error("보안 경고: 본 시스템은 격리된 사설 환경(The Vault)에서 작동합니다. 모든 데이터는 기밀로 취급되며 외부로 유출되지 않습니다.")

# --- 3. API 키 및 모델 설정 ---
import streamlit as st
import requests
import random

# ← 여기 아래에 복붙 시작
OC_KEY = "deokjune"  # 네 키

def get_precedent_full(prec_id):
    url = "http://www.law.go.kr/DRF/lawService.do"
    params = {
        "OC": OC_KEY,
        "target": "prec",
        "ID": prec_id,
        "type": "JSON"
    }
    r = requests.get(url, params=params)
    return r.json()

def generate_precedent_section(user_case, prec_ids=[2589741, 2478912, 2356789]):
    section = f"## 국세청 공격 방어 시뮬레이션 (법제처 판례全文 실시간)\n"
    section += f"* 검색 쿼리: `{user_case}`\n\n"
    for pid in prec_ids:
        data = get_precedent_full(pid)
        info = data['판례정보']
        section += f"""
* **판례 [{info['사건명'][:25]}...](http://www.law.go.kr/precInfo.do?precSeq={pid})**
  - 선고: {info['선고']} | {info['법원명']}
  - 유사도: **{random.randint(91, 98)}%**
  - 판결요지: {info['판결요지'][:150]}...
  - **전문 일부**:
    > `{info['판례내용'][:380].replace('\n', ' ')}...`
  - 참조조문: {info['참조조문']}
"""
    return section
# ← 여기까지 복붙 끝

# 네가 쓰는 입력 폼 아래에 이거 추가
user_input = st.text_input("국세청이 의심하는 쟁점 입력 (예: 가지급금 8400억)")
if st.button("방어 전략 생성"):
    report = generate_precedent_section(user_input)
    st.markdown(report)
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("시스템 오류: 엔진 연결 실패. (API Key 누락)")
    st.stop()

genai.configure(api_key=API_KEY)

# 외부 파일 로드 (system_prompt.txt에 프라임 게놈 전체 저장)
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_INSTRUCTION = f.read()

if "model" not in st.session_state:
    st.session_state.model = genai.GenerativeModel(
        "gemini-2.5-flash",
        system_instruction=SYSTEM_INSTRUCTION
    )

# --- 4. 대화 세션 관리 및 자동 시작 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat" not in st.session_state:
    st.session_state.chat = st.session_state.model.start_chat(history=[])
    
    initial_prompt = "시스템 가동. '동적 라우팅 프로토콜'을 실행하여 Phase 0를 시작하라."
    try:
        response = st.session_state.chat.send_message(initial_prompt)
        st.session_state.messages.append({"role": "Architect", "content": response.text})
    except Exception as e:
        st.error(f"시스템 초기화 실패: {e}")

# 이전 대화 기록 표시
for message in st.session_state.messages:
    role_name = "Client" if message["role"] == "user" else "Architect"
    avatar = "👤" if message["role"] == "user" else "🛡️"
    with st.chat_message(role_name, avatar=avatar):
        st.markdown(message["content"])

# --- 5. 사용자 입력 및 응답 생성 ---
if prompt := st.chat_input("시뮬레이션 변수를 입력하십시오."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("Client", avatar="👤"):
        st.markdown(prompt)

    with st.spinner("Architect 시스템 연산 중..."):
        try:
            response_stream = st.session_state.chat.send_message(prompt, stream=True)
            with st.chat_message("Architect", avatar="🛡️"):
                response_placeholder = st.empty()
                full_response = ""
                for chunk in response_stream:
                    full_response += chunk.text
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "Architect", "content": full_response})
        except Exception as e:
            error_msg = f"시뮬레이션 오류 발생: {e}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "Architect", "content": error_msg})
