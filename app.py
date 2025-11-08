# ======================================================
# 🛡️ Veritas Engine v7.1 (Stabilized Build)
# ======================================================
import streamlit as st
import google.generativeai as genai
import os
import requests
import re
import numpy as np

# ======================================================
# 1. SYSTEM INIT (Vault Mode)
# ======================================================
st.set_page_config(page_title="베리타스 엔진 7.1", page_icon="🛡️", layout="centered")

hide_ui = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {visibility: hidden;}
</style>
"""
st.markdown(hide_ui, unsafe_allow_html=True)

st.title("🧠 베리타스 엔진 v7.1")
st.error("보안 경고: 본 시스템은 격리된 사설 환경(The Vault)에서 작동합니다. 외부 유출 금지.")

# ======================================================
# 2. API KEY SETUP
# ======================================================
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("시스템 오류: 'GOOGLE_API_KEY' 누락. [Secrets] 탭을 확인하라.")
    st.stop()

genai.configure(api_key=API_KEY)

try:
    OC_KEY = st.secrets["LAW_API_KEY"]
except KeyError:
    OC_KEY = "DEOKJUNE"

# ======================================================
# 3. LAW.GO.KR API HANDLER
# ======================================================
def get_precedent_full(prec_id):
    """법제처 API에서 판례 전문을 가져온다."""
    if not OC_KEY or OC_KEY == "DEOKJUNE_FALLBACK":
        return {"error": "LAW_API_KEY 누락"}
    url = "http://www.law.go.kr/DRF/lawService.do"
    params = {"OC": OC_KEY, "target": "prec", "ID": prec_id, "type": "JSON"}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if '판례정보' not in data:
            return {"error": f"API 응답 오류: {data}"}
        return data
    except Exception as e:
        return {"error": f"API 호출 실패: {e}"}


def show_full_precedent(prec_id):
    """판례 전문 표시 포맷."""
    data = get_precedent_full(prec_id)
    if "error" in data:
        return f"**[API 오류]** (ID: {prec_id}) → {data['error']}"
    try:
        info = data.get('판례정보', {})
        title = info.get('사건명', 'N/A')
        verdict_date = info.get('선고일자', 'N/A')
        court = info.get('법원명', 'N/A')
        summary = info.get('판결요지', 'N/A').replace('\n', ' ')
        content = info.get('판례내용', 'N/A')[:500].replace('\n', ' ')
        ref = info.get('참조조문', 'N/A')
        return f"""
---
**🔍 판례 전문 전체**
- 사건명: {title}
- 선고일자: {verdict_date}
- 법원: {court}
- [법제처 바로가기](http://www.law.go.kr/precInfo.do?precSeq={prec_id})
**요지**: {summary}
**본문 (500자)**: {content}...
**참조조문**: {ref}
---
"""
    except Exception as e:
        return f"[데이터 처리 오류]: {e}"

# ======================================================
# 4. EMBEDDING ENGINE (게릴라 RAG)
# ======================================================
EMBED_MODEL = "models/text-embedding-004"

def embed_text(text, task_type="RETRIEVAL_DOCUMENT"):
    try:
        res = genai.embed_content(model=EMBED_MODEL, content=text, task_type=task_type)
        return res['embedding']
    except Exception as e:
        st.error(f"임베딩 실패: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_and_embed_precedents(file_path):
    """GitHub RAW 또는 로컬 txt 파일을 읽어 임베딩한다."""
    try:
        # 🔹 RAW 경로 자동 판별
        if file_path.startswith("http://") or file_path.startswith("https://"):
            st.info(f"GitHub RAW 경로 감지 ✅\n{file_path}")
            r = requests.get(file_path, timeout=10)
            if r.status_code != 200:
                raise FileNotFoundError(f"HTTP 응답 {r.status_code}")
            content = r.text
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
    except FileNotFoundError:
        st.warning(f"⚠️ 탄약고({file_path})를 찾지 못했습니다. GitHub 업로드를 확인하세요.")
        return [], np.array([])
    except Exception as e:
        st.error(f"탄약고 로드 실패: {e}")
        return [], np.array([])

    precedents = [p.strip() for p in content.split('---END OF PRECEDENT---') if p.strip()]
    if not precedents:
        st.warning(f"⚠️ 탄약고({file_path})가 비어 있습니다.")
        return [], np.array([])

    st.success(f"✅ 탄약고 장전 완료! 판례 {len(precedents)}개 확보.")
    embeddings = []
    valid_precedents = []
    for p in precedents:
        emb = embed_text(p)
        if emb:
            embeddings.append(emb)
            valid_precedents.append(p)
    return valid_precedents, np.array(embeddings)


def find_similar_precedents(query, precedents, embeddings, top_k=3):
    """입력 쿼리와 유사한 판례 반환"""
    if embeddings.size == 0:
        return ""
    q_emb = embed_text(query, task_type="RETRIEVAL_QUERY")
    if q_emb is None:
        return ""
    sims = np.dot(embeddings, q_emb)
    top = np.argsort(sims)[-top_k:][::-1]
    context = "\n\n[참조: 게릴라 RAG 유사 판례 탐색 결과]\n"
    for i in top:
        if sims[i] > 0.7:
            context += f"--- (유사도 {sims[i]*100:.0f}%)\n{precedents[i][:800]}...\n"
    return context

# ======================================================
# 5. SYSTEM PROMPT + 모델 초기화
# ======================================================
try:
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
except Exception:
    SYSTEM_PROMPT = "당신은 법률 AI 시스템 '베리타스 엔진'입니다."

if "precedents" not in st.session_state:
    RAW_URL = "https://raw.githubusercontent.com/deokjune85-rgb/imdmirage/main/precedents_data.txt"
    st.session_state.precedents, st.session_state.embeddings = load_and_embed_precedents(RAW_URL)

if "model" not in st.session_state:
    st.session_state.model = genai.GenerativeModel("gemini-2.5-flash", system_instruction=SYSTEM_PROMPT)

if "chat" not in st.session_state:
    st.session_state.chat = st.session_state.model.start_chat(history=[])
    st.session_state.messages = []

# ======================================================
# 6. UI (대화 인터페이스)
# ======================================================
for msg in st.session_state.messages:
    role = "Client" if msg["role"] == "user" else "Architect"
    avatar = "👤" if msg["role"] == "user" else "🛡️"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])

if prompt := st.chat_input("시뮬레이션 변수를 입력하십시오."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("Client", avatar="👤"):
        st.markdown(prompt)

    with st.spinner("Architect 연산 중..."):
        try:
            # 🔹 RAG 검색
            rag_context = find_similar_precedents(prompt, st.session_state.precedents, st.session_state.embeddings)
            full_prompt = prompt + rag_context

            # 🔹 생성 모델 호출
            response_stream = st.session_state.chat.send_message(full_prompt, stream=True)
            with st.chat_message("Architect", avatar="🛡️"):
                placeholder = st.empty()
                answer = ""
                for chunk in response_stream:
                    answer += chunk.text
                    placeholder.markdown(answer + "▌")
                placeholder.markdown(answer)

            # 🔹 법제처 API 판례 호출 자동 후처리
            if any(x in prompt for x in ["판례", "전문", "ID", "본문"]):
                ids = re.findall(r'\d{6,8}', prompt)
                for pid in ids[:3]:
                    with st.spinner(f"법제처 판례 {pid} 호출 중..."):
                        answer += "\n\n" + show_full_precedent(pid)
                placeholder.markdown(answer)

            st.session_state.messages.append({"role": "Architect", "content": answer})

        except Exception as e:
            st.error(f"시뮬레이션 오류: {e}")
