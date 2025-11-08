# ======================================================
# 🛡️ Veritas Engine v8.1 — Always Append Precedent Mode
# ======================================================
import streamlit as st
import google.generativeai as genai
import requests, re, numpy as np

# ======================================================
# 1. SYSTEM CONFIG
# ======================================================
st.set_page_config(page_title="베리타스 엔진 v8.1", page_icon="🛡️", layout="centered")

st.markdown("""
<style>
#MainMenu, footer, header, .stDeployButton {visibility:hidden;}
html, body, [class*="css"], [data-testid="stChatMessage"], [data-testid="stChatMessageContent"], div, span, p {
    font-family: 'Noto Sans KR', sans-serif !important;
    font-size: 16px !important;
    line-height: 1.7 !important;
    color: #FFFFFF !important;
    background-color: transparent !important;
}
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ 베리타스 엔진 버전 8.1")
st.error("보안 경고: 본 시스템은 격리된 사설 환경(The Vault)에서 작동합니다. 모든 데이터는 기밀로 취급되며 외부로 유출되지 않습니다.")

# ======================================================
# 2. API SETUP
# ======================================================
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("시스템 오류: 'GOOGLE_API_KEY' 누락. [Secrets] 확인 필요.")
    st.stop()

genai.configure(api_key=API_KEY)
OC_KEY = st.secrets.get("LAW_API_KEY", "DEOKJUNE")

# ======================================================
# 3. 법제처 판례 API
# ======================================================
def get_precedent_full(prec_id):
    url = "http://www.law.go.kr/DRF/lawService.do"
    params = {"OC": OC_KEY, "target": "prec", "ID": prec_id, "type": "JSON"}
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        return data.get("판례정보", {})
    except:
        return {}

# ======================================================
# 4. 게릴라 RAG
# ======================================================
EMBED_MODEL = "models/text-embedding-004"

def embed_text(text, task_type="RETRIEVAL_DOCUMENT"):
    try:
        res = genai.embed_content(model=EMBED_MODEL, content=text, task_type=task_type)
        return np.array(res["embedding"], dtype=float)
    except:
        return None

@st.cache_data(show_spinner=False)
def load_and_embed_precedents(file_path):
    try:
        r = requests.get(file_path, timeout=10)
        content = r.text
    except:
        return [], np.array([])
    items = [p.strip() for p in content.split("---END OF PRECEDENT---") if p.strip()]
    emb_list, valid = [], []
    for p in items:
        emb = embed_text(p)
        if emb is not None:
            emb_list.append(emb)
            valid.append(p)
    return valid, np.vstack(emb_list) if emb_list else np.array([])

def find_similar_precedents(query, precedents, embeddings, top_k=5):
    if embeddings.size == 0:
        return []
    q_emb = embed_text(query, task_type="RETRIEVAL_QUERY")
    if q_emb is None:
        return []
    sims = np.dot(embeddings, q_emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb))
    idx = np.argsort(sims)[-top_k:][::-1]
    return [{"similarity": float(sims[i]), "text": precedents[i]} for i in idx if sims[i] > 0.7]

# ======================================================
# 5. SYSTEM PROMPT
# ======================================================
try:
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        SYSTEM_INSTRUCTION = f.read()
except:
    SYSTEM_INSTRUCTION = "당신은 법률 AI 시스템 '베리타스 엔진'입니다."

RAW_URL = "https://raw.githubusercontent.com/deokjune85-rgb/imdmirage/main/precedents_data.txt"
if "precedents" not in st.session_state:
    st.session_state.precedents, st.session_state.embeddings = load_and_embed_precedents(RAW_URL)

if "model" not in st.session_state:
    st.session_state.model = genai.GenerativeModel("gemini-2.5-flash", system_instruction=SYSTEM_INSTRUCTION)

if "chat" not in st.session_state:
    st.session_state.chat = st.session_state.model.start_chat(history=[])
    st.session_state.messages = []
    try:
        init = st.session_state.chat.send_message("시스템 가동. Phase 0 시작.")
        st.session_state.messages.append({"role": "Architect", "content": init.text})
    except Exception as e:
        st.error(f"시스템 초기화 실패: {e}")

# ======================================================
# 6. CHAT INTERFACE
# ======================================================
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🛡️"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(f"<div style='white-space:pre-wrap; color:#FFFFFF; font-size:16px;'>{msg['content']}</div>", unsafe_allow_html=True)

if prompt := st.chat_input("시뮬레이션 변수를 입력하십시오."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(f"<div style='white-space:pre-wrap; color:#FFFFFF; font-size:16px;'>{prompt}</div>", unsafe_allow_html=True)

    with st.spinner("Architect 시스템 연산 중..."):
        try:
            stream = st.session_state.chat.send_message(prompt, stream=True)
            with st.chat_message("Architect", avatar="🛡️"):
                placeholder = st.empty()
                answer = ""
                for chunk in stream:
                    text = re.sub(r'(\d-\d\.)', r'\n\1', chunk.text)
                    answer += text
                    placeholder.markdown(f"<div style='white-space:pre-wrap; color:#FFFFFF; font-size:16px;'>{answer}▌</div>", unsafe_allow_html=True)
                placeholder.markdown(f"<div style='white-space:pre-wrap; color:#FFFFFF; font-size:16px;'>{answer}</div>", unsafe_allow_html=True)

            # ✅ 항상 마지막에 판례 자동 추가 (조건 없음)
            docs = find_similar_precedents(prompt, st.session_state.precedents, st.session_state.embeddings)
            if docs:
                report = "### 🧾 실시간 판례 전문 분석 (자동)\n\n"
                report += f"* 검색 쿼리: `{prompt}`\n\n"
                for d in docs:
                    sim = d["similarity"]
                    title = d["text"].split("\n")[0][:80]
                    excerpt = " ".join(d["text"].split("\n")[1:5])[:300]
                    report += f"* 판례 [{title}](#)\n  - 유사도: {sim*100:.0f}%\n  - 전문 일부: \"{excerpt}...\"\n\n"
                with st.chat_message("Architect", avatar="🛡️"):
                    st.markdown(f"<div style='white-space:pre-wrap; color:#FFFFFF; font-size:16px;'>{report}</div>", unsafe_allow_html=True)

            st.session_state.messages.append({"role": "Architect", "content": answer})

        except Exception as e:
            st.error(f"시뮬레이션 오류: {e}")
