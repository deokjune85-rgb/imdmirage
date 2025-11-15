# =====================================================
# 🛡️ 베리타스 엔진 8.0 — 실시간 판례 API + 법령 RAG (TXT)
# =====================================================
import streamlit as st
import google.generativeai as genai
import requests
import xml.etree.ElementTree as ET
import os
import re
import numpy as np
import time
import json

# --- 1. 시스템 설정 ---
st.set_page_config(page_title="베리타스 엔진 8.0", page_icon="🛡️", layout="centered")

custom_css = """
<style>
#MainMenu, footer, header, .stDeployButton {visibility:hidden;}
html, body, div, span, p {
    font-family: 'Noto Sans KR', sans-serif !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.title("🛡️ 베리타스 엔진 v8.0 — 실시간 판례 API 적용")
st.warning("※ 이 시스템은 The Vault 내부 전용입니다. 모든 데이터는 외부로 유출되지 않습니다.")

# --- 2. GOOGLE API KEY ---
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=API_KEY)
except:
    st.error("❌ GOOGLE_API_KEY 없음")
    st.stop()

# =====================================================
# ⚖️ 3. 법제처 판례 API 설정 (OC 기반)
# =====================================================
LAW_OC_ID = "deokjune"  # ← 이미 확인된 네 OC
LAW_SEARCH_URL = "https://www.law.go.kr/DRF/mobPrecSearch.do"  # ← 네가 XML 받은 엔드포인트

def search_precedents_from_api(keyword: str, page: int = 1, per_page: int = 20):
    """
    🔍 law.go.kr 판례 API 실시간 검색
    """
    try:
        params = {
            "OC": LAW_OC_ID,
            "keyword": keyword,      # ※ 일부 계정은 query=써야 할 수 있음
            "type": "XML",
            "page": page
        }

        resp = requests.get(LAW_SEARCH_URL, params=params, timeout=10)
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        results = []

        for prec in root.findall("prec"):
            pid = (prec.findtext("판례일련번호") or "").strip()
            title = (prec.findtext("사건명") or "").strip()
            case_no = (prec.findtext("사건번호") or "").strip()
            date = (prec.findtext("선고일자") or "").strip()
            court = (prec.findtext("법원명") or "").strip()
            detail = (prec.findtext("판례상세링크") or "").strip()

            if detail.startswith("http"):
                url = detail
            else:
                url = "https://www.law.go.kr" + detail

            rag_index = f"{court} {case_no} / {date}\n{title}\n원문: {url}"

            results.append({
                "id": pid,
                "title": title,
                "case_no": case_no,
                "court": court,
                "date": date,
                "url": url,
                "rag_index": rag_index,
                "similarity": 1.0,
            })

        return results

    except Exception as e:
        print(f"[판례 API 에러] {e}")
        return []


# =====================================================
# ⚖️ 4. 법령 RAG (statutes_data.txt)
# =====================================================
EMBEDDING_MODEL_NAME = "models/text-embedding-004"

def embed_text(text, task_type="retrieval_document"):
    try:
        clean = text.replace("\n", " ").strip()
        if not clean:
            return None
        res = genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=clean,
            task_type=task_type
        )
        return res["embedding"]
    except:
        return None

@st.cache_data(show_spinner=True)
def load_statutes(path="statutes_data.txt", separator=r"\s*---END OF STATUTE---\s*", max_items=300):
    """
    txt 법령 로더 → 문단별 임베딩
    """
    if not os.path.exists(path):
        return [], []

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = re.split(separator, content)
    chunks = [c.strip() for c in chunks if c.strip()][:max_items]

    items = []
    emb = []

    for c in chunks:
        e = embed_text(c)
        if e:
            items.append({"rag_index": c})
            emb.append(e)

    print(f"[법령 RAG] {len(items)}개 로드됨")
    return items, emb


# =====================================================
# ⚙️ 5. 시스템 프롬프트 로드
# =====================================================
try:
    SYSTEM_PROMPT = open("system_prompt.txt", encoding="utf-8").read()
except:
    st.error("❌ system_prompt.txt 없음")
    st.stop()


# =====================================================
# 🤖 6. 모델 초기화
# =====================================================
if "model" not in st.session_state:
    st.session_state.model = genai.GenerativeModel(
        "models/gemini-2.5-flash",
        system_instruction=SYSTEM_PROMPT
    )

    # 법령만 로드 (판례는 API로 실시간 검색)
    with st.spinner("법령 RAG 초기화 중..."):
        S_DATA, S_EMB = load_statutes()
        st.session_state.statutes = S_DATA
        st.session_state.s_embeddings = S_EMB

    st.session_state.chat = st.session_state.model.start_chat(history=[])
    st.session_state.messages = []


# =====================================================
# 🧠 7. 유사 법령 검색
# =====================================================
def find_similar_items(query_text, items, embeddings, top_k=3, threshold=0.75):
    if not items or not embeddings:
        return []
    q_emb = embed_text(query_text, "retrieval_query")
    if q_emb is None:
        return []

    sims = np.dot(np.array(embeddings), np.array(q_emb))
    idxs = np.argsort(sims)[::-1][:top_k]

    results = []
    for i in idxs:
        if float(sims[i]) >= threshold:
            d = items[i].copy()
            d["similarity"] = float(sims[i])
            results.append(d)

    return results


# =====================================================
# 💬 8. 메시지 출력
# =====================================================
for m in st.session_state.messages:
    avatar = "👤" if m["role"] == "user" else "🛡️"
    with st.chat_message(avatar):
        st.markdown(m["content"])


# =====================================================
# 🧠 9. 메인 입력 처리
# =====================================================
if prompt := st.chat_input("사건 설명 또는 질문을 입력하십시오."):
    # --- 사용자 메시지 표시 ---
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("👤"):
        st.markdown(prompt)

    rag_context = ""

    # 1) 법령 RAG
    statutes_found = find_similar_items(
        prompt,
        st.session_state.statutes,
        st.session_state.s_embeddings,
        top_k=3,
        threshold=0.75
    )
    if statutes_found:
        rag_context += "\n\n[관련 법령]\n"
        for s in statutes_found:
            rag_context += f"- {s['rag_index'][:200]}...\n"

    # 2) 판례 API (OC)
    with st.spinner("판례 API 검색 중..."):
        api_cases = search_precedents_from_api(prompt, page=1, per_page=10)

    if api_cases:
        rag_context += "\n\n[실시간 유사 판례]\n"
        for c in api_cases[:5]:
            rag_context += f"- {c['rag_index']}\n\n"

    # --- Gemini 프롬프트 ---
    final_prompt = f"{prompt}\n\n{rag_context}"

    # --- 모델 응답 ---
    with st.chat_message("🛡️"):
        res = st.session_state.chat.send_message(final_prompt)
        st.markdown(res.text)
        st.session_state.messages.append({"role": "assistant", "content": res.text})
