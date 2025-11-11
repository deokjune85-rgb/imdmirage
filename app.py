# ======================================================
# 🛡️ 베리타스 엔진 7.6 — Contextual Dual RAG (JSONL/TXT Hybrid) + Relay Mechanism
# ======================================================
import streamlit as st
import google.generativeai as genai
import os
import numpy as np
import re
import time
import json # ★ JSONL 처리를 위한 모듈 추가

# --- 1. 시스템 설정 및 CSS (기존 7.5 버전 유지) ---
st.set_page_config(page_title="베리타스 엔진 7.6", page_icon="🛡️", layout="centered")

custom_css = """
<style>
#MainMenu, footer, header, .stDeployButton {visibility:hidden;}
html, body, div, span, p {
    font-family: 'Noto Sans KR', sans-serif !important;
    font-size: 16px !important;
    line-height: 1.7 !important;
}
h1 { text-align: left !important; font-weight: 900 !important; font-size: 36px !important; margin-top: 10px !important; margin-bottom: 15px !important; }
strong, b { font-weight: 700; }
.fadein { animation: fadeInText 0.5s ease-in-out forwards; opacity: 0; }
@keyframes fadeInText { from {opacity: 0; transform: translateY(3px);} to {opacity: 1; transform: translateY(0);} }
[data-testid="stChatMessageContent"] { font-size: 16px !important; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- 2. 타이틀 및 경고 ---
st.title("베리타스 엔진 버전 7.6")
st.warning("보안 경고: 본 시스템은 격리된 사설 환경(The Vault)에서 작동합니다. 모든 데이터는 기밀로 취급되며 외부로 유출되지 않습니다.")

# --- 3. API 키 및 RAG 엔진 설정 ---
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
    if not API_KEY:
         raise ValueError("API Key is empty.")
    genai.configure(api_key=API_KEY)
except (KeyError, ValueError) as e:
    st.error(f"시스템 오류: 엔진 연결 실패. {e}")
    st.stop()

# --- [RAG 엔진 함수 정의] (★핵심 수정: JSONL/TXT 하이브리드 로더★) ---
EMBEDDING_MODEL_NAME = "models/text-embedding-004"

def embed_text(text, task_type="retrieval_document"):
    try:
        clean_text = text.replace('\n', ' ').strip()
        if not clean_text: return None
        result = genai.embed_content(model=EMBEDDING_MODEL_NAME, content=clean_text, task_type=task_type)
        return result['embedding']
    except Exception as e:
        print(f"Embedding error: {e}"); return None

# [★핵심 수정★] 통합 데이터 로더 (JSONL 및 TXT 지원)
@st.cache_data
def load_and_embed_data(file_path, separator_regex=None):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}"); return [], []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}"); return [], []

    if not content.strip(): return [], []

    data_items, embeddings = [], []

    # JSONL 파일 처리 (.jsonl 확장자)
    if file_path.endswith('.jsonl'):
        for line in content.strip().split('\n'):
            try:
                item = json.loads(line)
                # 'rag_index' 필드를 임베딩 (핵심!)
                text_to_embed = item.get('rag_index')
                if text_to_embed:
                    ebd = embed_text(text_to_embed, task_type="retrieval_document")
                    if ebd:
                        embeddings.append(ebd)
                        # 전체 객체(원문, 링크 포함)를 저장
                        data_items.append(item)
            except json.JSONDecodeError:
                continue
    
    # TXT 파일 처리 (법령 데이터 및 하위 호환성)
    elif separator_regex:
        chunks = re.split(separator_regex, content)
        raw_items = [p.strip() for p in chunks if p and p.strip()]
        for item_text in raw_items:
            ebd = embed_text(item_text, task_type="retrieval_document")
            if ebd:
                embeddings.append(ebd)
                # TXT는 구조화되지 않았으므로 텍스트 자체를 객체화하여 저장 (구조 통일)
                data_items.append({"rag_index": item_text, "raw_text": item_text})

    print(f"[RAG] Loaded {len(data_items)} items from {file_path}.")
    return data_items, embeddings

# 검색 함수 (기존 유지)
def find_similar_items(query_text, items, embeddings, top_k=3, threshold=0.50):
    if not embeddings or not items: return []
    q_emb = embed_text(query_text, task_type="retrieval_query")
    if q_emb is None: return []
    
    sims = np.dot(np.array(embeddings), np.array(q_emb))
    idxs = np.argsort(sims)[::-1][:top_k]
    
    results = []
    for i in idxs:
        if float(sims[i]) >= threshold:
            # 결과에 전체 객체와 유사도를 저장 (이미 객체화되어 있음)
            result_item = items[i].copy()
            result_item["similarity"] = float(sims[i])
            results.append(result_item)
    return results


# (유틸리티 함수 유지 - 생략)
def _is_menu_input(s: str) -> bool: ...
def _is_final_report(txt: str) -> bool: ...
def _query_title(prompt_text: str) -> str: ...
def update_active_module(response_text): ...

# --- 4. 시스템 프라임 유전자 (Prime Genome) 로드 및 초기화 ---
try:
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        SYSTEM_INSTRUCTION = f.read()
    # ... (검증 로직 생략) ...
except (FileNotFoundError, ValueError) as e:
    st.error(f"치명적 오류: 시스템 코어(system_prompt.txt) 로드 실패. {e}")
    st.stop()


if "model" not in st.session_state:
    try:
        st.session_state.model = genai.GenerativeModel("models/gemini-2.5-flash",
                                                    system_instruction=SYSTEM_INSTRUCTION)
        
        # [★수정됨★] 듀얼 RAG 초기화 (JSONL + TXT)
        with st.spinner("분석 엔진(Dual RAG) 초기화 중... (최초 실행 시)"):
            # 1. 판례 데이터 로드 (P-RAG) - JSONL 우선, TXT 폴백
            p_data, p_emb = load_and_embed_data('precedents_data.jsonl')
            if not p_data:
                 # JSONL이 없거나 비었으면 TXT 시도
                 p_data, p_emb = load_and_embed_data('precedents_data.txt', r'\s*---END OF PRECEDENT---\s*')

            st.session_state.precedents = p_data
            st.session_state.p_embeddings = p_emb

            # 2. 법령 데이터 로드 (S-RAG) - TXT 방식 유지
            s_data, s_emb = load_and_embed_data('statutes_data.txt', r'\s*---END OF STATUTE---\s*')
            st.session_state.statutes = s_data
            st.session_state.s_embeddings = s_emb
        
        st.session_state.active_module = "초기 상태 (미정의)"

    except Exception as e:
        st.error(f"시스템 초기화 실패: {e}")
        st.stop()

# --- 5, 6. 대화 세션 관리 및 출력 (기존 유지) ---
# ... (생략) ...

# --- 7. 입력 및 응답 생성 (★핵심 수정: JSONL 기반 출력 및 릴레이★) ---

# 스트리밍 출력 함수 (기존 유지)
def stream_and_store_response(chat_session, prompt_to_send, spinner_text="Architect 시스템 연산 중..."):
    # ... (함수 내용 유지 - 생략) ...

# 메인 입력 루프
if prompt := st.chat_input("시뮬레이션 변수를 입력하십시오."):
    # ... (사용자 입력 처리 및 Phase 2 감지 생략) ...

    # Contextual RAG 실행
    rag_context = ""
    similar_precedents = []
    
    if not _is_menu_input(prompt):
        # ... (Contextual Query 생성 생략) ...

        with st.spinner("실시간 데이터베이스 분석 중... (Dual RAG: 판례/법령)..."):
            # 1. 법령 검색 (S-RAG)
            if ("statutes" in st.session_state and st.session_state.statutes):
                similar_statutes = find_similar_items(contextual_query, st.session_state.statutes, st.session_state.s_embeddings, top_k=3, threshold=0.75)
                if similar_statutes:
                    # LLM 주입용 텍스트 생성 ('rag_index' 사용)
                    s_texts = [f"[유사도: {c['similarity']:.2f}]\n{c.get('rag_index', '내용 없음')}\n---\n" for c in similar_statutes]
                    rag_context += "\n\n[시스템 참조: 검색된 관련 법령 데이터]\n" + "\n".join(s_texts)

            # 2. 판례 검색 (P-RAG)
            if ("precedents" in st.session_state and st.session_state.precedents):
                similar_precedents = find_similar_items(contextual_query, st.session_state.precedents, st.session_state.p_embeddings, top_k=5, threshold=0.75)
                if similar_precedents:
                    # LLM 주입용 텍스트 생성 ('rag_index' 사용)
                    p_texts = [f"[유사도: {c['similarity']:.2f}]\n{c.get('rag_index', '내용 없음')}\n---\n" for c in similar_precedents]
                    rag_context += "\n\n[시스템 참조: 검색된 유사 판례 데이터]\n" + "\n".join(p_texts)

    # 최종 프롬프트 구성 및 시스템 응답 생성
    final_prompt = f"{prompt}\n{rag_context}"
    current_response = stream_and_store_response(st.session_state.chat, final_prompt)

    # 릴레이 메커니즘 (기존 로직 유지)
    # ... (생략) ...

    # [★핵심 수정★] 판례 시각화 및 원문 보기 기능 (JSONL 기반)
    clean_response = re.sub('<[^<]+?>', '', current_response)
    if _is_final_report(clean_response) and similar_precedents:
        q_title = _query_title(prompt)
        st.markdown("**📚 실시간 판례 전문 분석 (P-RAG 결과)**\n\n* 검색 쿼리: `[" + q_title + "]`\n")

        for case_data in similar_precedents[:3]:
            # JSONL 객체에서 메타데이터 추출
            sim_pct = int(round(case_data["similarity"] * 100))
            title = case_data.get('title', '제목 없음')
            case_no = case_data.get('case_no', case_data.get('id', ''))
            court = case_data.get('court', '')
            date = case_data.get('date', '')
            url = case_data.get('url')
            full_text = case_data.get('full_text', case_data.get('raw_text')) # 전문 또는 TXT 폴백
            
            label = f"판례 [{title}]"
            if court and case_no:
                label += f" — {court} {case_no}"

            # 요약 카드 출력 ('rag_index' 사용)
            summary = case_data.get('rag_index', '요약 내용 없음')
            if len(summary) > 200: summary = summary[:197] + "..."

            # 링크 생성
            action_link = f"[🔗 원문 링크 보기]({url})" if url else ""

            item_md = (
                f"* **{label}**\n"
                f"  - 선고: {date} | 유사도: {sim_pct}% | {action_link}\n"
                f"  - 내용 요약 (RAG Index): {summary}"
            )
            st.markdown(item_md)
            
            # [★신설★] 원문 보기 기능 (Expander 사용)
            if full_text:
                with st.expander("📄 판례 전문 보기"):
                    # 전문은 텍스트 형식으로 출력 (가독성 확보)
                    st.text(full_text)

    elif _is_final_report(clean_response) and not _is_menu_input(prompt) and not similar_precedents:
         st.info("ℹ️ 분석과 관련된 유사 판례가 데이터베이스에서 검색되지 않았습니다. (임계값 0.75)")
