import streamlit as st
import google.generativeai as genai
import os
import requests
import re

# --- 1. 시스템 설정 ---
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

st.title("베리타스 엔진 버전 7.0")
st.error("보안 경고: 본 시스템은 격리된 사설 환경에서 작동합니다.")

# --- 2. API 키 설정 ---
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("GOOGLE_API_KEY 누락")
    st.stop()
genai.configure(api_key=API_KEY)

# --- 3. 법제처 API (완전 새로 작성) ---
OC_KEY = "deokjune"  # 네 키

def get_precedent_full(prec_id):
    url = "http://www.law.go.kr/DRF/lawService.do"
    params = {"OC": OC_KEY, "target": "prec", "ID": prec_id, "type": "JSON"}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if '판례정보' not in data:
            return {"error": "판례 없음"}
        return data
    except:
        return {"error": "호출 실패"}

def show_full_precedent(prec_id):
    data = get_precedent_full(prec_id)
    if "error" in data:
        return f"\n---\n[판례 호출 실패] ID: {prec_id}\n{data['error']}\n---"
    info = data['판례정보']
    return f"""
---
[법제처 실시간 판례 전문]

사건명: {info.get('사건명', 'N/A')}
선고: {info.get('선고', 'N/A')} | 법원: {info.get('법원명', 'N/A')}
링크: http://www.law.go.kr/precInfo.do?precSeq={prec_id}

판결요지:
{info.get('판결요지', 'N/A')[:300]}...

전문 일부 (500자):
{info.get('판례내용', 'N/A')[:500].replace('\n', ' ')}

참조조문:
{info.get('참조조문', 'N/A')}
---
"""

# --- 4. 시스템 프롬프트 로드 ---
try:
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        SYSTEM_INSTRUCTION = f.read()
except:
    st.error("system_prompt.txt 없음")
    st.stop()

if "model" not in st.session_state:
    st.session_state.model = genai.GenerativeModel("gemini-2.5-flash", system_instruction=SYSTEM_INSTRUCTION)

# --- 5. 세션 초기화 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat" not in st.session_state:
    st.session_state.chat = st.session_state.model.start_chat(history=[])
    st.session_state.chat.send_message("시스템 가동. Phase 0 시작.")

# --- 6. 이전 메시지 출력 ---
for msg in st.session_state.messages:
    role = "Client" if msg["role"] == "user" else "Architect"
    avatar = "👤" if msg["role"] == "user" else "🛡️"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])

# --- 7. 입력 및 응답 ---
if prompt := st.chat_input("입력하십시오"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("Client", avatar="👤"):
        st.markdown(prompt)

    with st.spinner("연산 중..."):
        try:
            response_stream = st.session_state.chat.send_message(prompt, stream=True)
            with st.chat_message("Architect", avatar="🛡️"):
                placeholder = st.empty()
                full = ""
                for chunk in response_stream:
                    full += chunk.text
                    placeholder.markdown(full + "▌")
                
                # 판례 요청 자동 감지 및 삽입
                if any(x in prompt.lower() for x in ["판례", "전문", "본문", "id"]):
                    ids = re.findall(r'\d{6,8}', prompt)
                    if ids:
                        with st.spinner(f"법제처에서 판례 {len(ids[:3])}개 호출 중..."):
                            for pid in ids[:3]:
                                full += "\n\n" + show_full_precedent(pid)
                
                placeholder.markdown(full)
            
            st.session_state.messages.append({"role": "Architect", "content": full})
        except Exception as e:
            st.error(f"오류: {e}")
