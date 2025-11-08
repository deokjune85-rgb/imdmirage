# ======================================================
# 🛡️ Veritas Engine v7.8 — Full Dark Clean Mode
# ======================================================
import streamlit as st
import google.generativeai as genai
import requests, re, os, numpy as np

# ======================================================
# 1. SYSTEM CONFIG
# ======================================================
st.set_page_config(page_title="베리타스 엔진 v7.8", page_icon="🛡️", layout="centered")

# === 💡 전역 디자인 CSS (글자색 흰색 + 사이즈 통일 + 마크다운 정렬) ===
st.markdown("""
<style>
#MainMenu, footer, header, .stDeployButton {visibility:hidden;}
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', sans-serif !important;
    font-size: 16px !important;
    line-height: 1.7 !important;
    color: #FFFFFF !important;
    background-color: #0E1117 !important;
}
[data-testid="stChatMessageContent"] {
    font-family: 'Noto Sans KR', sans-serif !important;
    font-size: 16px !important;
    color: #FFFFFF !important;
    line-height: 1.7 !important;
    white-space: pre-wrap !important;
}
h1, h2, h3, h4, h5, h6 {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}
.stMarkdown p {
    color: #FFFFFF !important;
    font-size: 16px !important;
    line-height: 1.7 !important;
}
hr {border: none !important; border-top: 1px solid #444 !important;}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ 베리타스 엔진 버전 7.8")
st.error("보안 경고: 본 시스템은 격리된 사설 환경(The Vault)에서 작동합니다. 모든 데이터는 기밀로 취급되며 외부로 유출되지 않습니다.")

# ======================================================
# 2. API KEYS
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
# 3. 법제처 API 자판기
# ======================================================
def get_precedent_full(prec_id):
    url = "http://www.law.go.kr/DRF/lawService.do"
    params = {"OC": OC_KEY, "target": "prec", "ID": prec_id, "type": "JSON"}
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if '판례정보' not in data:
            return {"error": "API 응답 구조 이상"}
        return data
    except Exception as e:
        return {"error": f"API 실패: {e}"}

def show_full_precedent(prec_id):
    data = get_precedent_full(prec_id)
    if "error" in data:
        return f"---\n**[API 분석 실패]** (ID: {prec_id})\n{data['error']}\n---"
    try:
        info = data.get('판례정보', {})
        title = info.get('사건명', 'N/A')
        verdict = info.get('선고일자', 'N/A')
        court = info.get('법원명', 'N/A')
        summary = info.get('판결요지', 'N/A').replace('\n',' ')
        body = info.get('판례내용','N/A')[:500].replace('\n',' ')
        ref = info.get('참조조문','N/A')
        return f"""
---
**🔍 판례 전문 전체 (법제처 실시간 호출)**
**사건명**: {title}
**선고**: {verdict} | **법원**: {court}
**판례 링크**: [법제처 바로가기](http://www.law.go.kr/precInfo.do?precSeq={prec_id})
**판결요지**: {summary}
**전문 일부 (500자)**: {body}...
**참조조문**: {ref}
---
"""
    except Exception as e:
        return f"---\n[API 분석 실패]: {e}\n---"

# ======================================================
# 4. 게릴라 RAG (탄약고)
# ======================================================
EMBED_MODEL = "models/text-embedding-004"

def embed_text(text, task_type="RETRIEVAL_DOCUMENT"):
    try:
        res = genai.embed_content(model=EMBED_MODEL, content=text, task_type=task_type)
        return np.array(res['embedding'], dtype=float)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def load_and_embed_precedents(file_path):
    try:
        if file_path.startswith("http"):
            r = requests.get(file_path, timeout=10)
            r.raise_for_status()
            content = r.text
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
    except Exception:
        return [], np.array([])

    precedents = [p.strip() for p in content.split('---END OF PRECEDENT---') if p.strip()]
    if not precedents:
        return [], np.array([])

    emb_list, valid = [], []
    for p in precedents:
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
    emb_norms = np.linalg.norm(embeddings, axis=1)
    q_norm = np.linalg.norm(q_emb)
    sims = np.dot(embeddings, q_emb) / (emb_norms * q_norm)
    top_k_idx = np.argsort(sims)[-top_k:][::-1]
    selected_docs = []
    for i in top_k_idx:
        sim = sims[i]
        text = precedents[i]
        selected_docs.append({"similarity": float(sim), "text": text})
    return selected_docs

# ======================================================
# 5. SYSTEM PROMPT
# ======================================================
try:
    with open("system_prompt.txt","r",encoding="utf-8") as f:
        SYSTEM_INSTRUCTION = f.read()
except Exception:
    SYSTEM_INSTRUCTION = "당신은 법률 AI 시스템 '베리타스 엔진'입니다."

RAW_URL = "https://raw.githubusercontent.com/deokjune85-rgb/imdmirage/main/precedents_data.txt"

if "precedents" not in st.session_state:
    st.session_state.precedents, st.session_state.embeddings = load_and_embed_precedents(RAW_URL)

if "model" not in st.session_state:
    st.session_state.model = genai.GenerativeModel("gemini-2.5-flash", system_instruction=SYSTEM_INSTRUCTION)

if "chat" not in st.session_state:
    st.session_state.chat = st.session_state.model.start_chat(history=[])
    st.session_state.messages = []
    initial_prompt = "시스템 가동. '동적 라우팅 프로토콜'을 실행하여 Phase 0를 시작하라."
    try:
        response = st.session_state.chat.send_message(initial_prompt)
        st.session_state.messages.append({"role": "Architect", "content": response.text})
    except Exception as e:
        st.error(f"시스템 초기화 실패: {e}")

# ======================================================
# 6. UI / CHAT
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

    with st.spinner("Architect 시스템 연산 중..."):
        try:
            response_stream = st.session_state.chat.send_message(prompt, stream=True)
            with st.chat_message("Architect", avatar="🛡️"):
                placeholder = st.empty()
                answer = ""
                for chunk in response_stream:
                    # === 자동 줄바꿈 처리 (Phase 구문 정리) ===
                    chunk_text = chunk.text.replace("2-1.", "\n2-1.").replace("2-2.", "\n2-2.").replace("2-3.", "\n2-3.").replace("2-4.", "\n2-4.").replace("2-5.", "\n2-5.").replace("2-6.", "\n2-6.").replace("2-7.", "\n2-7.").replace("2-8.", "\n2-8.")
                    answer += chunk_text
                    placeholder.markdown(f"<div style='white-space:pre-wrap; color:#FFFFFF; font-size:16px; line-height:1.7;'>{answer}▌</div>", unsafe_allow_html=True)
                placeholder.markdown(f"<div style='white-space:pre-wrap; color:#FFFFFF; font-size:16px; line-height:1.7;'>{answer}</div>", unsafe_allow_html=True)

            # ======================================================
            # ✅ Phase-End 판례 자동 후처리 (최종 출력 시)
            # ======================================================
            if (
                any(kw in answer for kw in ["최종", "보고서", "브리핑", "결과 요약", "완료"])
                and not any(kw in answer for kw in ["입력", "Phase", "단계", "시작"])
            ):
                selected_docs = find_similar_precedents(
                    prompt, st.session_state.precedents, st.session_state.embeddings
                )
                if selected_docs:
                    report_md = "### 🧾 실시간 판례 전문 분석 (최종)\n\n"
                    report_md += f"* 검색 쿼리: `{prompt}`\n\n"
                    for doc in selected_docs:
                        sim = doc["similarity"]
                        text = doc["text"]
                        lines = text.split('\n')
                        title = lines[0][:80] if lines else "제목 없음"
                        excerpt = " ".join(lines[1:5])[:300].strip()
                        report_md += (
                            f"* 판례 [{title}](#)\n"
                            f"  - 유사도: {sim*100:.0f}%\n"
                            f"  - 전문 일부: \"{excerpt}...\"\n\n"
                        )
                    with st.chat_message("Architect", avatar="🛡️"):
                        st.markdown(
                            f"<div style='font-family:Noto Sans KR; color:#FFFFFF; font-size:16px; line-height:1.7;'>{report_md}</div>",
                            unsafe_allow_html=True
                        )
                    st.session_state.messages.append({"role": "Architect", "content": report_md})

            st.session_state.messages.append({"role": "Architect", "content": answer})

        except Exception as e:
            st.error(f"시뮬레이션 오류: {e}")
