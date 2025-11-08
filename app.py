import streamlit as st
import google.generativeai as genai
import os # 'system_prompt.txt'를 '열기' 위한 '필수' 모듈
import requests # 네놈이 '요청'한 '용병(API)' 모듈
import re # 네놈이 '요청'한 '트리거(Trigger)' 모듈

import streamlit as st
import google.generativeai as genai
import os
import requests  # 이미 있음
import re       # 이미 있음

# ← 여기 아래에 이거 딱 붙여라 (OC_KEY만 네 키로 바꿔!)
OC_KEY = "deokjune"  # ← 여기만 "deokjune" → 네 실제 OC 값으로 바꿔!

def get_precedent_full(prec_id):
    url = "http://www.law.go.kr/DRF/lawService.do"
    params = {
        "OC": OC_KEY,
        "target": "prec",
        "ID": prec_id,
        "type": "JSON"
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except:
        return {"error": "API 호출 실패"}

def show_full_precedent(prec_id):
    data = get_precedent_full(prec_id)
    if "error" in data:
        return f"---\n**[판례 호출 실패]** ID: {prec_id}\n{data['error']}\n---"
    try:
        info = data['판례정보']
        return f"""
---
**법제처 실시간 판례 전문 (ID: {prec_id})**

**사건명**: {info.get('사건명', 'N/A')}
**선고**: {info.get('선고', 'N/A')} | **법원**: {info.get('법원명', 'N/A')}
**판례 바로가기**: [법제처 링크](http://www.law.go.kr/precInfo.do?precSeq={prec_id})

**판결요지**  
{info.get('판결요지', 'N/A')}

**전문 일부 (500자)**  

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

try:
    # '구글' API 키 '약탈'
    API_KEY = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("시스템 오류: 'GOOGLE_API_KEY' '약탈' 실패. 'Secrets'를 '확인'하라.")
    st.stop()

genai.configure(api_key=API_KEY)

# --- ★★★ 법제처 API 연동 (네놈의 '용병' 코드) ★★★ ---
try:
    OC_KEY = st.secrets["LAW_API_KEY"]
except KeyError:
    OC_KEY = "DEOKJUNE_FALLBACK"

def get_precedent_full(prec_id):
    """
    법제처 API를 호출하여 판례 ID(prec_id)로 판례 전문을 '실시간'으로 '약탈'한다.
    """
    if OC_KEY == "DEOKJUNE_FALLBACK":
        return {"error": "[치명적 오류]: 'LAW_API_KEY' '약탈' 실패. [st.secrets]를 '확인'하라."}
        
    url = "http://www.law.go.kr/DRF/lawService.do"
    params = {
        "OC": OC_KEY,
        "target": "prec",
        "ID": prec_id,
        "type": "JSON"
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status() 
        data = r.json()
        if '판례정보' not in data:
             return {"error": f"법제처 API 오류: {data.get('Error', '알 수 없는 응답')}"}
        return data
    except requests.exceptions.RequestException as e:
        return {"error": f"API 호출 실패: {e}"}

def show_full_precedent(prec_id):
    """
    '약탈'한 '판례(JSON)'를 'EPE'가 '이해'할 수 있는 '텍스트'로 '재가공(Formatting)'한다.
    """
    data = get_precedent_full(prec_id)
    if "error" in data:
        return f"--- \n**[API 분석 실패]** (ID: {prec_id})\n{data['error']}\n---"
    
    # --- ★★★ '오류' '수정' 지점 (Try Block) ★★★ ---
    try:
        info = data.get('판례정보', {})
        if not info:
             return f"--- \n**[API 분석 실패]** (ID: {prec_id})\n'판례정보' 필드를 '데이터'에서 '식별'할 수 없음.\n---"

        prec_id_display = info.get('판례일련번호', prec_id)
        title = info.get('사건명', 'N/A')
        verdict_date = info.get('선고일자', 'N/A')
        court_name = info.get('법원명', 'N/A')
        summary = info.get('판결요지', 'N/A').replace(chr(10), ' ') 
        full_text = info.get('판례내용', 'N/A')[:500].replace(chr(10), ' ')
        ref_law = info.get('참조조문', 'N/A').replace(chr(10), ' ')
        
        return f"""
---
**🔍 판례 전문 전체 (법제처 실시간 호출)**
**사건명**: {title}
**선고**: {verdict_date} | **법원**: {court_name}
**판례 링크**: [법제처 바로가기](http://www.law.go.kr/precInfo.do?precSeq={prec_id_display})

**판결요지**:
{summary}

**전문 일부 (500자)**:
{full_text}...

**참조조문**:
{ref_law}
---
"""
    # --- ★★★ '수정'된 'Except' 구문 ★★★ ---
    except Exception as e:
        return f"--- \n**[API 분석 실패]** (ID: {prec_id})\n'데이터' '가공' 중 '치명적 오류' 발생: {e}\n---"
# --- ★★★ 법제처 API 이식 종료 ★★★ ---


# 외부 파일 로드 (system_prompt.txt에 프라임 게놈 전체 저장)
try:
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        SYSTEM_INSTRUCTION = f.read()
except FileNotFoundError:
    st.error("'system_prompt.txt' 파일을 '약탈'하는 데 '실패'했다, 이 머저리야. '파일'을 '업로드'해.")
    st.stop()
except Exception as e:
    st.error(f"시스템 프롬프트 로드 '실패': {e}")
    st.stop()


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

# --- 5. 사용자 입력 및 응답 생성 (★수정된 교리★) ---
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
                    response_placeholder.markdown(full_response + "▌") # 타이핑 효과
                
                # --- ★★★ 법제처 API 연동 (네놈의 '용병' 코드) ★★★ ---
                if any(x in prompt.lower() for x in ["판례", "전문", "본문", "판결문", "전체", "아이디"]):
                    ids = re.findall(r'\d{6,8}', prompt) # 6~8자리 숫자를 'ID'로 '간주'
                    if ids:
                        with st.spinner(f"법제처 API 호출... 판례 ID {', '.join(ids)} '실시간 약탈' 중..."):
                            for pid in ids[:3]:  # 최대 3개 '약탈'
                                precedent_text = show_full_precedent(pid)
                                full_response += "\n\n" + precedent_text
                
                response_placeholder.markdown(full_response) 
            
            st.session_state.messages.append({"role": "Architect", "content": full_response})
        
        except Exception as e:
            error_msg = f"시뮬레이션 오류 발생: {e}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "Architect", "content": error_msg})
