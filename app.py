# 베리타스 엔진 8.1 — Domain 메뉴 개선 + Dual RAG (TXT/JSONL 하이브리드)
# ======================================================

import streamlit as st
import google.generativeai as genai
import os
import re
import json
import numpy as np
import time

# ---------------------------------------
# 0. 기본 세팅
# ---------------------------------------
st.set_page_config(
    page_title="베리타스 엔진 8.1",
    page_icon="🛡️",
    layout="centered"
)

# CSS 스타일
custom_css = """
<style>
#MainMenu, footer, header, .stDeployButton {visibility:hidden;}
.stChatMessage {border-radius:10px; padding:10px; margin-bottom:10px;}
.stChatMessage[data-testid="user"] {background:#e8f4f8;}
.stChatMessage[data-testid="assistant"] {background:#f0f0f0;}
.precedent-card {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 12px;
    margin: 8px 0;
    background: #fafafa;
}
.precedent-card h4 {
    margin: 0 0 8px 0;
    color: #1f77b4;
}
.precedent-card .similarity {
    display: inline-block;
    background: #4CAF50;
    color: white;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.85em;
    margin-bottom: 6px;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 상단 타이틀 + 경고
st.title("베리타스 엔진 8.1")
st.caption("Phase 0: 도메인 선택 → 이후 Architect가 자동 라우팅")

st.warning(
    "⚠️ **법률 자문 면책**: 본 시스템의 분석 결과는 법률 자문이 아니며, "
    "실제 법률 사건에는 반드시 변호사와 상담하시기 바랍니다."
)

# ---------------------------------------
# 1. API 키 설정
# ---------------------------------------
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ GOOGLE_API_KEY가 설정되지 않았습니다. Streamlit Cloud Secrets에서 설정하세요.")
    st.stop()

genai.configure(api_key=api_key)

# ---------------------------------------
# 2. 임베딩 / RAG 유틸
# ---------------------------------------
EMBEDDING_MODEL_NAME = "models/text-embedding-004"


def embed_text(text: str, task_type: str = "retrieval_document"):
    """텍스트를 임베딩 벡터로 변환"""
    clean_text = text.replace("\n", " ").strip()
    if not clean_text:
        return None
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=clean_text,
            task_type=task_type
        )
        return result["embedding"]
    except Exception as e:
        print(f"[Embedding error] {e}")
        return None


@st.cache_data(show_spinner=True)
def load_and_embed_data(file_path: str, separator_regex: str = None):
    """
    데이터 파일을 로드하고 임베딩 생성
    - .jsonl: 줄 단위 JSON -> item['rag_index']를 임베딩
    - .txt: separator_regex 기준으로 분할하여 임베딩
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

    # JSONL 파일 처리
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
    
    # TXT 파일 처리
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
    """쿼리와 유사한 항목 검색"""
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
# 3. 각종 유틸 함수
# ---------------------------------------
def _is_menu_input(s: str) -> bool:
    """메뉴 선택 입력인지 확인 (예: 1, 2-3)"""
    return bool(re.fullmatch(r"^\s*\d{1,2}(?:-\d{1,2})?\s*$", s))


def _is_final_report(txt: str) -> bool:
    """최종 보고서인지 확인"""
    return "전략 브리핑 보고서" in txt


def _query_title(prompt_text: str) -> str:
    """쿼리 제목 생성 (70자 제한)"""
    return prompt_text[:67] + "..." if len(prompt_text) > 70 else prompt_text


def update_active_module(response_text: str):
    """응답 텍스트에서 활성화된 모듈 추출 및 업데이트"""
    m = re.search(r"'(.*?)' 모듈을 (?:최종 )?활성화합니다", response_text)
    if m:
        st.session_state.active_module = m.group(1).strip()
    elif "Phase 0" in response_text and not st.session_state.get("active_module"):
        st.session_state.active_module = "Phase 0 (도메인 선택)"


# ---------------------------------------
# 4. 시스템 프라임 프롬프트 로드
# ---------------------------------------
try:
    with open("system_instruction.txt", "r", encoding="utf-8") as f:
        SYSTEM_INSTRUCTION = f.read()
except FileNotFoundError:
    st.error("system_instruction.txt 파일을 찾을 수 없습니다.")
    st.stop()

# ---------------------------------------
# 5. Phase 0 — 도메인 선택 UI
# ---------------------------------------
DEFAULT_OPTION = "선택 안 함 (자동 판단)"
domain_options = [
    DEFAULT_OPTION,
    "형사",
    "민사",
    "가사/이혼",
    "행정",
    "노동",
    "부동산",
    "지적재산",
    "조세",
    "기타(혼합)",
]

# 세션 상태 초기화
if "selected_domain" not in st.session_state:
    st.session_state.selected_domain = DEFAULT_OPTION

st.subheader("Phase 0 — 사건 도메인 선택")

# 라디오 버튼
selected_domain = st.radio(
    "현재 사건이 속한 주 도메인을 선택하세요. (선택 안 함 시 시스템이 자동으로 판단합니다.)",
    domain_options,
    index=domain_options.index(st.session_state.selected_domain),
    horizontal=True,
)

st.session_state.selected_domain = selected_domain
st.info(f"현재 도메인 설정: **{selected_domain}**")

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
        st.error(f"시스템 초기화 실패 (모델 로드 오류): {e}")
        st.stop()

    st.session_state.messages = []
    st.session_state.active_module = f"Phase 0 — {selected_domain}"

    # RAG 코퍼스 지연 로딩 설정
    st.session_state.precedents = []
    st.session_state.p_embeddings = []
    st.session_state.statutes = []
    st.session_state.s_embeddings = []

    # 초기 인사/배치
    try:
        domain_info = selected_domain
        if selected_domain == DEFAULT_OPTION:
            domain_info = "미정의 (시스템 자동 판단 필요)"

        init_prompt = (
            f"시스템 가동. 현재 설정된 도메인: {domain_info}. "
            f"Phase 0에서 사건 구조를 스캔하고, 이후 Phase 1~를 동적으로 라우팅하라. "
            f"만약 도메인이 미정의라면, 사용자의 첫 입력을 분석하여 최적의 도메인을 판단하고 활성화하라."
        )
        resp = st.session_state.chat.send_message(init_prompt)
        init_text = resp.text

        st.session_state.messages.append({"role": "user", "content": "(시스템 부팅)"})
        st.session_state.messages.append({"role": "Architect", "content": init_text})

        update_active_module(init_text)
    except Exception as e:
        st.error(f"초기화 중 오류 발생: {e}")

# ---------------------------------------
# 7. 과거 메시지 렌더링
# ---------------------------------------
for m in st.session_state.messages:
    role_name = "Client" if m["role"] == "user" else "Architect"
    avatar_icon = "👤" if m["role"] == "user" else "🛡️"
    with st.chat_message(role_name, avatar=avatar_icon):
        st.markdown(m["content"], unsafe_allow_html=True)

# ---------------------------------------
# 8. 스트리밍 응답 함수
# ---------------------------------------
def stream_and_store_response(chat_session, prompt_to_send: str,
                               spinner_text: str = "Architect 시스템 연산 중..."):
    """LLM 응답을 스트리밍으로 받아서 표시하고 저장"""
    full_response = ""
    start_time = time.time()

    with st.chat_message("Architect", avatar="🛡️"):
        placeholder = st.empty()
        try:
            with st.spinner(spinner_text):
                stream = chat_session.send_message(prompt_to_send, stream=True)
                for chunk in stream:
                    # 응답 유효성 검사
                    if not getattr(chunk, "parts", None) or not getattr(chunk, "text", None):
                        if not full_response:
                            full_response = "[시스템 경고: 응답 생성 실패 또는 안전 필터에 의해 차단됨.]"
                            placeholder.error(full_response)
                            break
                    full_response += chunk.text
                    placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                placeholder.markdown(full_response, unsafe_allow_html=True)
        except Exception as e:
            full_response = f"[치명적 오류: {e}]"
            placeholder.error(full_response)

    st.session_state.messages.append({"role": "Architect", "content": full_response})
    update_active_module(full_response)

    end_time = time.time()
    print(f"[LLM] 응답 시간: {end_time - start_time:.2f}s")
    return full_response


# ---------------------------------------
# 9. 메인 입력 루프 + Dual RAG
# ---------------------------------------
if prompt := st.chat_input("사건 정보 또는 질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("Client", avatar="👤"):
        st.markdown(prompt, unsafe_allow_html=True)

    # Phase 상태 확인
    is_data_ingestion_phase = "Phase 2" in (st.session_state.active_module or "")

    # RAG 코퍼스 최초 로딩
    if not st.session_state.statutes and not st.session_state.precedents:
        with st.spinner("시스템 데이터베이스 초기화 중..."):
            # 법령 JSONL -> 없으면 TXT 폴백
            s_data, s_emb = load_and_embed_data("statutes_data.jsonl")
            if not s_data:
                s_data, s_emb = load_and_embed_data(
                    "statutes_data.txt",
                    r"\s*---END OF STATUTE---\s*",
                )
            st.session_state.statutes = s_data
            st.session_state.s_embeddings = s_emb

            # 판례 JSONL -> 없으면 TXT 폴백
            p_data, p_emb = load_and_embed_data("precedents_data.jsonl")
            if not p_data:
                p_data, p_emb = load_and_embed_data(
                    "precedents_data.txt",
                    r"\s*---END OF PRECEDENT---\s*",
                )
            st.session_state.precedents = p_data
            st.session_state.p_embeddings = p_emb

    # RAG 검색 (메뉴 입력이 아니고, Phase 2가 아닐 때)
    rag_context = ""
    similar_precedents = []

    if not _is_menu_input(prompt) and not is_data_ingestion_phase:
        current_domain = st.session_state.selected_domain
        if current_domain == DEFAULT_OPTION:
            current_domain = "미정의 (자동 판단 중)"

        contextual_query = (
            f"현재 활성화된 모듈: {st.session_state.active_module}. "
            f"선택된 도메인: {current_domain}. "
            f"사용자 질문/사실관계: {prompt}"
        )

        with st.spinner("실시간 데이터베이스 분석 중... (Dual RAG: 법령/판례)"):
            # 법령 검색
            if st.session_state.statutes:
                s_hits = find_similar_items(
                    contextual_query,
                    st.session_state.statutes,
                    st.session_state.s_embeddings,
                    top_k=3,
                    threshold=0.75,
                )
                if s_hits:
                    s_texts = []
                    for hit in s_hits:
                        txt = hit.get("rag_index") or hit.get("raw_text", "")
                        s_texts.append(f"- {txt} (유사도: {hit['similarity']:.2f})")
                    rag_context += (
                        "\n\n[관련 법령 데이터베이스 검색 결과]\n" +
                        "\n".join(s_texts)
                    )

            # 판례 검색
            if st.session_state.precedents:
                similar_precedents = find_similar_items(
                    contextual_query,
                    st.session_state.precedents,
                    st.session_state.p_embeddings,
                    top_k=3,
                    threshold=0.75,
                )
                if similar_precedents:
                    p_texts = []
                    for prec in similar_precedents:
                        txt = prec.get("rag_index") or prec.get("summary", "")
                        p_texts.append(f"- {txt} (유사도: {prec['similarity']:.2f})")
                    rag_context += (
                        "\n\n[관련 판례 데이터베이스 검색 결과]\n" +
                        "\n".join(p_texts)
                    )

    # 최종 프롬프트 구성
    current_domain = st.session_state.selected_domain
    if current_domain == DEFAULT_OPTION:
        current_domain = "미정의 (시스템 자동 판단 필요)"

    final_prompt = (
        f"[현재 설정된 도메인] {current_domain}\n"
        f"[사용자 원문 입력]\n{prompt}\n"
        f"{rag_context}"
    )

    # 시스템 응답 생성
    current_response = stream_and_store_response(
        st.session_state.chat,
        final_prompt,
    )

    # 판례 카드 시각화
    clean_response = re.sub("<[^<]+?>", "", current_response)

    if _is_final_report(clean_response) and similar_precedents:
        st.subheader("📚 참고 판례 요약")
        for idx, prec in enumerate(similar_precedents, start=1):
            case_number = prec.get("case_number", f"판례 {idx}")
            summary = prec.get("summary", prec.get("rag_index", "요약 없음"))
            court = prec.get("court", "법원 정보 없음")
            date = prec.get("date", "날짜 정보 없음")
            similarity = prec.get("similarity", 0.0)
            full_text = prec.get("full_text", "전문 없음")

            card_html = f"""
            <div class="precedent-card">
                <h4>📖 {case_number}</h4>
                <span class="similarity">유사도: {similarity:.1%}</span>
                <p><strong>법원:</strong> {court}</p>
                <p><strong>선고일:</strong> {date}</p>
                <p><strong>요지:</strong> {summary[:300]}...</p>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

            if full_text and full_text != "전문 없음":
                with st.expander("📄 판례 전문 보기"):
                    st.text(full_text)

    elif _is_final_report(clean_response) and not similar_precedents and not _is_menu_input(prompt):
        st.info(
            "ℹ️ 분석과 관련된 유사 판례가 데이터베이스에서 검색되지 않았습니다. "
            "(임계값 0.75)"
        )
