# 베리타스 엔진 8.0 — Domain 메뉴 + Dual RAG (TXT/JSONL 하이브리드)

import streamlit as st
import google.generativeai as genai
import os
import numpy as np
import re
import time
import json

# ---------------------------------------
# 0. 기본 세팅
# ---------------------------------------
st.set_page_config(
    page_title="베리타스 엔진 8.0",
    page_icon="🛡️",
    layout="centered"
)

custom_css = """
<style>
#MainMenu, footer, header, .stDeployButton {visibility:hidden;}
html, body, div, span, p {
    font-family: 'Noto Sans KR', sans-serif !important;
    font-size: 16px !important;
    line-height: 1.7 !important;
}
h1 {
    text-align: left !important;
    font-weight: 900 !important;
    font-size: 36px !important;
    margin-top: 10px !important;
    margin-bottom: 15px !important;
}
strong, b { font-weight: 700; }
.fadein { animation: fadeInText 0.5s ease-in-out forwards; opacity: 0; }
@keyframes fadeInText {
    from {opacity: 0; transform: translateY(3px);}
    to {opacity: 1; transform: translateY(0);}
}
[data-testid="stChatMessageContent"] {
    font-size: 16px !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 상단 타이틀 + 경고
st.title("베리타스 엔진 8.0")
st.caption("Phase 0: 도메인 선택 → 이후 Architect가 자동 라우팅")

st.warning(
    "보안 경고: 본 시스템은 격리된 사설 환경(The Vault)에서 작동합니다. "
    "모든 데이터는 기밀로 취급되며 외부로 유출되지 않습니다."
)

# ---------------------------------------
# 1. API 키 설정
# ---------------------------------------
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
    if not API_KEY:
        raise ValueError("API Key is empty.")
    genai.configure(api_key=API_KEY)
except (KeyError, ValueError) as e:
    st.error(f"시스템 오류: 엔진 연결 실패. {e}")
    st.stop()

# ---------------------------------------
# 2. 임베딩 / RAG 유틸
# ---------------------------------------
EMBEDDING_MODEL_NAME = "models/text-embedding-004"


def embed_text(text: str, task_type: str = "retrieval_document"):
    clean_text = text.replace("\n", " ").strip()
    if not clean_text:
        return None
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=clean_text,
            task_type=task_type,
        )
        return result["embedding"]
    except Exception as e:
        print(f"[Embedding error] {e}")
        return None


@st.cache_data(show_spinner=True)
def load_and_embed_data(file_path: str, separator_regex: str = None):
    """
    - .jsonl: 줄 단위 JSON ➜ item['rag_index']를 임베딩
    - .txt  : separator_regex 기준으로 쪼개서 임베딩
    """
    if not os.path.exists(file_path):
        print(f"[RAG] File not found: {file_path}")
        return [], []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"[RAG] Error reading file: {e}")
        return [], []

    if not content.strip():
        return [], []

    data_items = []
    embeddings = []

    # JSONL
    if file_path.endswith(".jsonl"):
        for line in content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            txt = obj.get("rag_index") or obj.get("summary") or ""
            if not txt:
                continue

            emb = embed_text(txt, task_type="retrieval_document")
            if emb:
                data_items.append(obj)
                embeddings.append(emb)

    # TXT
    elif separator_regex:
        parts = re.split(separator_regex, content)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            emb = embed_text(p, task_type="retrieval_document")
            if emb:
                data_items.append({"rag_index": p, "raw_text": p})
                embeddings.append(emb)

    print(f"[RAG] Loaded {len(data_items)} items from {file_path}")
    return data_items, embeddings


def find_similar_items(query_text, items, embeddings, top_k=3, threshold=0.5):
    if not items or not embeddings:
        return []

    q_emb = embed_text(query_text, task_type="retrieval_query")
    if q_emb is None:
        return []

    sims = np.dot(np.array(embeddings), np.array(q_emb))
    idxs = np.argsort(sims)[::-1][:top_k]

    results = []
    for i in idxs:
        score = float(sims[i])
        if score < threshold:
            continue
        item = items[i].copy()
        item["similarity"] = score
        results.append(item)

    return results


# ---------------------------------------
# 3. 각종 유틸 함수 (Phase 판단 등)
# ---------------------------------------
def _is_menu_input(s: str) -> bool:
    return bool(re.fullmatch(r"^\s*\d{1,2}(?:-\d{1,2})?\s*$", s))


def _is_final_report(txt: str) -> bool:
    return "전략 브리핑 보고서" in txt


def _query_title(prompt_text: str) -> str:
    return prompt_text[:67] + "..." if len(prompt_text) > 70 else prompt_text


def update_active_module(response_text: str):
    m = re.search(r"\[(.+?)\]' 모듈을 활성화합니다", response_text)
    if m:
        st.session_state.active_module = m.group(1).strip()
    elif "Phase 0" in response_text and not st.session_state.get("active_module"):
        st.session_state.active_module = "Phase 0 (도메인 선택)"


# ---------------------------------------
# 4. 시스템 프라임 프롬프트 로드
# ---------------------------------------
try:
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        SYSTEM_INSTRUCTION = f.read()
    if len(SYSTEM_INSTRUCTION) < 100:
        raise ValueError("System prompt is too short.")
except (FileNotFoundError, ValueError) as e:
    st.error(f"치명적 오류: system_prompt.txt 로드 실패. {e}")
    st.stop()

# ---------------------------------------
# 5. Phase 0 — 도메인 선택 UI (여기가 '선택지')
# ---------------------------------------
domain_options = [
    "형사",
    "민사",
    "가사/이혼",
    "파산·회생",
    "행정/조세",
    "회사·M&A",
    "의료/산재",
    "IP·저작권",
    "기타(혼합)",
]

if "selected_domain" not in st.session_state:
    st.session_state.selected_domain = "형사"

st.subheader("Phase 0 — 사건 도메인 선택")

selected_domain = st.radio(
    "현재 사건이 속한 주 도메인을 선택하세요.",
    domain_options,
    index=domain_options.index(st.session_state.selected_domain),
    horizontal=True,
)

st.session_state.selected_domain = selected_domain
st.info(f"현재 도메인: **{selected_domain}**")

# ---------------------------------------
# 6. 모델 & 세션 초기화
# ---------------------------------------
if "model" not in st.session_state:
    try:
        st.session_state.model = genai.GenerativeModel(
            "models/gemini-2.5-flash",
            system_instruction=SYSTEM_INSTRUCTION,
        )
        st.session_state.chat = st.session_state.model.start_chat(history=[])
    except Exception as e:
        st.error(f"시스템 초기화 실패: {e}")
        st.stop()

    st.session_state.messages = []
    st.session_state.active_module = f"Phase 0 — {selected_domain}"

    # RAG 코퍼스는 '지연 로딩' (처음 질문 들어올 때)
    st.session_state.precedents = []
    st.session_state.p_embeddings = []
    st.session_state.statutes = []
    st.session_state.s_embeddings = []

    # 초기 인사/배치
    try:
        init_prompt = (
            f"시스템 가동. 현재 선택된 도메인: {selected_domain}. "
            f"Phase 0에서 사건 구조를 스캔하고, 이후 Phase 1~를 동적으로 라우팅하라."
        )
        resp = st.session_state.chat.send_message(init_prompt)
        init_text = resp.text
    except Exception as e:
        init_text = f"[시스템 초기화 실패: {e}]"

    st.session_state.messages.append({"role": "Architect", "content": init_text})
    update_active_module(init_text)

# ---------------------------------------
# 7. 과거 메시지 렌더링
# ---------------------------------------
for m in st.session_state.messages:
    role_name = "Client" if m["role"] == "user" else "Architect"
    avatar = "👤" if m["role"] == "user" else "🛡️"
    with st.chat_message(role_name, avatar=avatar):
        st.markdown(m["content"], unsafe_allow_html=True)

# ---------------------------------------
# 8. 스트리밍 응답 함수
# ---------------------------------------
def stream_and_store_response(chat_session, prompt_t_
