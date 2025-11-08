import streamlit as st
import google.generativeai as genai
import os 
import requests 
import re 
import numpy as np 

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
    API_KEY = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("시스템 오류: 'GOOGLE_API_KEY' '약탈' 실패. 'Secrets'를 '확인'하라.")
    st.stop()

genai.configure(api_key=API_KEY)

# --- ★★★ '탄약고 B': 법제처 API (자판기) ★★★ ---
try:
    OC_KEY = st.secrets["LAW_API_KEY"]
except KeyError:
    OC_KEY = "DEOKJUNE_FALLBACK"

def get_precedent_full(prec_id):
    if OC_KEY == "DEOKJUNE_FALLBACK":
        return {"error": "[치명적 오류]: 'LAW_API_KEY' '약탈' 실패. [st.secrets]를 '확인'하라."}
    url = "http://www.law.go.kr/DRF/lawService.do"
    params = {"OC": OC_KEY, "target": "prec", "ID": prec_id, "type": "JSON"}
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
    data = get_precedent_full(prec_id)
    if "error" in data:
        return f"--- \n**[API 분석 실패]** (ID: {prec_id})\n{data['error']}\n---"
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
**판결요지**: {summary}
**전문 일부 (500자)**: {full_text}...
**참조조문**: {ref_law}
---
"""
    except Exception as e:
        return f"--- \n**[API 분석 실패]** (ID: {prec_id})\n'데이터' '가공' 중 '치명적 오류' 발생: {e}\n---"
# --- (법제처 API 종료) ---


# --- ★★★ '탄약고 A': 게릴라 RAG (트로이 목마) ★★★ ---
EMBEDDING_MODEL_NAME = "models/text-embedding-004" 

def embed_text(text, task_type="RETRIEVAL_DOCUMENT"):
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=text,
            task_type=task_type)
        return result['embedding']
    except Exception as e:
        st.error(f"임베딩 '오류' (모델 '호출' '실패'): {e}")
        return None

@st.cache_data(show_spinner=False)
def load_and_embed_precedents(file_path='precedents_data.txt'):
    """'txt' '쓰레기'를 '읽어' '벡터' '탄약'으로 '주조'한다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        st.warning(f"경고: '탄약고({file_path})' '발견' '실패'. '게릴라 RAG'가 '작동'하지 '않는다'.")
        # --- ★★★ '오류' '수정' (v4.1) ★★★ ---
        # '3개'가 '아니라' '2개'의 '쓰레기'를 '반환'한다.
        return [], np.array([])
    except Exception as e:
        st.error(f"'탄약고' '로드' '실패': {e}")
        return [], np.array([]) # '2개' '반환'

    precedents = content.split('---END OF PRECEDENT---')
    precedents = [p.strip() for p in precedents if p.strip()]
    
    if not precedents:
        st.warning(f"경고: '탄약고({file_path})'가 '비어'있다. '사기극' '실패'.")
        return [], np.array([]) # '2개' '반환'

    st.success(f"'{file_path}' '탄약고' '장전' '완료'. '총알(판례)' {len(precedents)}개 '확인'.")
    embeddings = []
    valid_precedents = []
    for p in precedents:
        emb = embed_text(p)
        if emb:
            embeddings.append(emb)
            valid_precedents.append(p)
    
    # '총알(텍스트)'과 '인식표(벡터)'를 '반환'한다.
    return valid_precedents, np.array(embeddings)

def find_similar_precedents(query_text, precedents, embeddings, top_k=3):
    """'사건'과 '가장' '유사한' '총알' 3개를 '발사'한다."""
    if embeddings.size == 0:
        return "" # '탄약고'가 '비었'다.

    query_embedding = embed_text(query_text, task_type="RETRIEVAL_QUERY")
    if query_embedding is None:
        return ""

    similarities = np.dot(embeddings, query_embedding)
    
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    
    context = "\n\n[시스템 참조: '게릴라 RAG'가 '탄약고(txt)'에서 '유사 판례' '탐지' '완료']\n"
    for i in top_k_indices:
        if similarities[i] > 0.7: 
            context += f"--- (유사도: {similarities[i]*100:.0f}%)\n{precedents[i]}\n---\n"
            
    return context
# --- ★★★ 게릴라 RAG 이식 종료 ★★★ ---


# --- '뇌(EPE)'와 '탄약고' '로딩' ---
try:
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        SYSTEM_INSTRUCTION = f.read()
except FileNotFoundError:
    st.error("'system_prompt.txt' 파일을 '약탈'하는 데 '실패'했다, 이 머저리야. '파일'을 '업로드'해.")
    st.stop()

# '탄약고 A(RAG)' '장전' (앱 '시작' 시 '1회' '실행')
# --- ★★★ '오류' '수정' 지점 (v4.1) ★★★ ---
# '171번' '라인'이 '여기'다. 'load_and_embed_precedents'는 '이제' '2개'만 '반환'한다.
if "precedents" not in st.session_state:
    st.session_state.precedents, st.session_state.embeddings = load_and_embed_precedents()

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

# --- 5. 사용자 입력 및 응답 생성 (★궁극의 융합 교리★) ---
if prompt := st.chat_input("시뮬레이션 변수를 입력하십시오."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("Client", avatar="👤"):
        st.markdown(prompt)

    with st.spinner("Architect 시스템 연산 중..."):
        try:
            # --- ★ 1. '게릴라 RAG' '선제' '발사' ★ ---
            with st.spinner("'탄약고 A(txt)'에서 '유사 판례' '탐색' 중..."):
                rag_context = find_similar_precedents(
                    prompt, 
                    st.session_state.precedents, 
                    st.session_state.embeddings
                )
            
            final_prompt_to_epe = prompt + rag_context

            # --- ★ 2. '뇌(EPE)' '작동' ★ ---
            response_stream = st.session_state.chat.send_message(final_prompt_to_epe, stream=True)
            
            with st.chat_message("Architect", avatar="🛡️"):
                response_placeholder = st.empty()
                full_response = ""
                for chunk in response_stream:
                    full_response += chunk.text
                    response_placeholder.markdown(full_response + "▌") 
                
                # --- ★ 3. '자판기(API)' '후처리' ★ ---
                if any(x in prompt.lower() for x in ["판례", "전문", "본문", "판결문", "전체", "아이디"]):
                    ids = re.findall(r'\d{6,8}', prompt) 
                    if ids:
                        with st.spinner(f"법제처 API 호출... 판례 ID {', '.join(ids)} '실시간 약탈' 중..."):
                            for pid in ids[:3]:
                                precedent_text = show_full_precedent(pid)
                                full_response += "\n\n" + precedent_text
                
                response_placeholder.markdown(full_response) 
            
            st.session_state.messages.append({"role": "Architect", "content": full_response})
        
        except Exception as e:
            error_msg = f"시뮬레이션 오류 발생: {e}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "Architect", "content": error_msg})
