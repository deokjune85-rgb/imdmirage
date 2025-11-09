# ======================================================
# 🛡️ 베리타스 엔진 7.2 — Dual RAG Build (Omega-Infinitum Core)
# ======================================================
import streamlit as st
import google.generativeai as genai
import os
import numpy as np
import re
import time # 속도 조절을 위한 모듈

# --- 1. 시스템 설정 (The Vault & Mirage Protocol) ---
# 테마 설정: 시스템 기본값 사용 (흰 바탕/검은 글씨 또는 다크 모드 자동 호환)
st.set_page_config(page_title="베리타스 엔진 7.2", page_icon="🛡️", layout="centered")

# CSS 해킹 (신기루 프로토콜) - [★수정됨: 색상 강제 제거, 애니메이션 최적화]
custom_css = """
<style>
#MainMenu, footer, header, .stDeployButton {visibility:hidden;}

/* --- 글자 스타일 통일 (색상 강제 제거) --- */
html, body, div, span, p {
    font-family: 'Noto Sans KR', sans-serif !important;
    /* color: #FFFFFF !important; <-- 이 쓰레기가 문제였다. 제거함. */
    font-size: 16px !important;
    line-height: 1.7 !important;
}

/* --- 타이틀 위치 조정 --- */
h1 {
    text-align: left !important;
    font-weight: 900 !important;
    font-size: 36px !important;
    margin-top: 10px !important;
    margin-bottom: 15px !important;
}

/* --- 중요 문단 / 헤드라인 강조 --- */
strong, b {
    font-weight: 700;
}

/* --- 부드러운 텍스트 등장 (속도 조절 0.5s) --- */
.fadein {
    animation: fadeInText 0.5s ease-in-out forwards;
    opacity: 0;
}
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


# --- 2. 타이틀 및 경고 ---
st.title("베리타스 엔진 버전 7.2")
# 라이트 모드에서는 st.error보다 st.warning이 가독성이 좋음.
st.warning("보안 경고: 본 시스템은 격리된 사설 환경(The Vault)에서 작동합니다. 모든 데이터는 기밀로 취급되며 외부로 유출되지 않습니다.")


# --- 3. API 키 및 모델 설정 ---
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
    if not API_KEY:
         raise ValueError("API Key is empty.")
    genai.configure(api_key=API_KEY)
except (KeyError, ValueError) as e:
    st.error(f"시스템 오류: 엔진 연결 실패. {e}")
    st.stop()

# --- [작전명: 듀얼 RAG 엔진] 함수 정의 (일반화) ---
EMBEDDING_MODEL_NAME = "models/text-embedding-004"

def embed_text(text, task_type="retrieval_document"):
    try:
        clean_text = text.replace('\n', ' ').strip()
        if not clean_text: return None
        result = genai.embed_content(model=EMBEDDING_MODEL_NAME, content=clean_text, task_type=task_type)
        return result['embedding']
    except Exception as e:
        print(f"Embedding error: {e}"); return None

# [★수정됨★] 데이터 로드 및 임베딩 함수 (일반화)
@st.cache_data
def load_and_embed_data(file_path, separator_regex):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}"); return [], []
    try:
        with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}"); return [], []

    if not content.strip(): return [], []

    # 정규식 기반 분할
    chunks = re.split(separator_regex, content)
    data_items = [p.strip() for p in chunks if p and p.strip()]

    embeddings, valid_items = [], []
    for item in data_items:
        ebd = embed_text(item, task_type="retrieval_document")
        if ebd:
            embeddings.append(ebd); valid_items.append(item)
    print(f"[RAG] Loaded {len(valid_items)} items from {file_path}.")
    return valid_items, embeddings

# [★수정됨★] 검색 함수 일반화 (판례/법령 공용)
def find_similar_items(query_text, items, embeddings, top_k=3, threshold=0.50):
    if not embeddings or not items: return []
    q_emb = embed_text(query_text, task_type="retrieval_query")
    if q_emb is None: return []
    
    sims = np.dot(np.array(embeddings), np.array(q_emb))
    idxs = np.argsort(sims)[::-1][:top_k]
    
    results = []
    for i in idxs:
        if float(sims[i]) >= threshold:
            results.append({"similarity": float(sims[i]), "raw_text": items[i]})
    return results

# (판례 파싱 함수는 시각화를 위해 유지)
def _parse_precedent_block(text: str) -> dict:
    # (기존 파싱 함수 내용 유지 - 생략)
    t = text.strip()
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    title = lines[0][:120] if lines else "제목 없음"
    m = re.search(r'\[(?P<court>[^ \[\]]+)\s+(?P<date>\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.)\s*선고\s*(?P<caseno>\d{4}\s*[가-힣]{1,2}\s*\d{3,6})\s*판결\]', t)
    court, date, caseno = (m.group('court'), m.group('date'), m.group('caseno').replace(" ", "")) if m else ("", "", "")
    if not caseno:
        m2 = re.search(r'(?P<caseno>\d{4}\s*[가-힣]{1,2}\s*\d{3,6})', t)
        if m2: caseno = m2.group('caseno').replace(" ", "")
    holding = ""
    m2 = re.search(r'【판결요지】(.*?)(【|$)', t, re.S)
    if m2: holding = re.sub(r'\s+', ' ', m2.group(1)).strip()
    else:
        m3 = re.search(r'【판시사항】(.*?)(【|$)', t, re.S)
        if m3: holding = re.sub(r'\s+', ' ', m3.group(1)).strip()
    if not holding: holding = re.sub(r'\s+', ' ', t)[:160].strip()
    excerpt = ""
    for key in ["【전문】", "【이 유】", "【이유】", "【본문】"]:
        pos = t.find(key)
        if pos != -1:
            excerpt = re.sub(r'\s+', ' ', t[pos:pos+300]).strip(); break
    if not excerpt: excerpt = re.sub(r'\s+', ' ', t)[:300].strip()
    if len(holding) > 130: holding = holding[:130].rstrip() + "…"
    if len(excerpt) > 160: excerpt = excerpt[:160].rstrip() + "…"
    return {"title": title, "court": court, "date": date, "case_no": caseno, "holding": holding, "excerpt": excerpt}

# (유틸리티 함수 유지)
def _is_menu_input(s: str) -> bool:
    if not s: return False
    return bool(re.fullmatch(r'\d+|[1-9]-\d+', s.strip()))

def _is_final_report(txt: str) -> bool:
    if not txt: return False
    t = txt.replace(" ", "")
    hits = 0
    for key in ["전략브리핑보고서", "리스크시뮬레이션분석", "권장다음단계", "면책조항"]:
        if key in t: hits += 1
    return (hits >= 2) and (len(t) > 500)

def _query_title(prompt_text: str) -> str:
    # (기존 함수 내용 유지)
    if not prompt_text: return ""
    m = re.search(r'\[([^\]]+)\]', prompt_text)
    if m: return m.group(1).strip()
    first = prompt_text.strip().splitlines()[0].strip()
    return (first[:77] + "…") if len(first) > 80 else first

# --- 4. 시스템 프라임 유전자 (Prime Genome) 로드 및 초기화 ---
try:
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        SYSTEM_INSTRUCTION = f.read()
    if not SYSTEM_INSTRUCTION.strip():
        raise ValueError("System prompt file is empty.")
except (FileNotFoundError, ValueError) as e:
    st.error(f"치명적 오류: 시스템 코어(system_prompt.txt) 로드 실패. {e}")
    st.stop()


if "model" not in st.session_state:
    try:
        # 모델명 확인: 'models/gemini-2.5-flash'
        st.session_state.model = genai.GenerativeModel("models/gemini-2.5-flash",
                                                    system_instruction=SYSTEM_INSTRUCTION)
        
        # [★수정됨★] 듀얼 RAG 초기화
        with st.spinner("분석 엔진(Dual RAG) 초기화 중... (최초 실행 시)"):
            # 1. 판례 데이터 로드 (P-RAG)
            p_data, p_emb = load_and_embed_data('precedents_data.txt', r'\s*---END OF PRECEDENT---\s*')
            st.session_state.precedents = p_data
            st.session_state.p_embeddings = p_emb
            if not p_data: st.warning("⚠️ 판례 데이터 로드 실패. P-RAG 비활성화.")

            # 2. 법령 데이터 로드 (S-RAG)
            s_data, s_emb = load_and_embed_data('statutes_data.txt', r'\s*---END OF STATUTE---\s*')
            st.session_state.statutes = s_data
            st.session_state.s_embeddings = s_emb
            if not s_data: st.warning("⚠️ 법령 데이터 로드 실패. S-RAG 비활성화.")

    except Exception as e:
        st.error(f"시스템 초기화 실패: {e}")
        st.stop()

# --- 5. 대화 세션 관리 및 자동 시작 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat" not in st.session_state or not st.session_state.messages:
    if "model" in st.session_state:
        try:
            if "chat" not in st.session_state:
                st.session_state.chat = st.session_state.model.start_chat(history=[])

            if not st.session_state.messages:
                initial_prompt = "긴급 명령: EPE 활성화. 즉시 <KnowledgeBase>의 'Phase 0: 도메인 선택 프로토콜'을 실행하고 메뉴를 출력하라. 다른 설명이나 확인은 생략한다."
                response = st.session_state.chat.send_message(initial_prompt)
                if response and response.text:
                     st.session_state.messages.append({"role": "Architect", "content": f"<div class='fadein'>{response.text}</div>"})
                else:
                     st.error("시스템 코어 응답 실패 (응답 없음).")
        except Exception as e:
            st.error(f"시스템 초기화 실패 (API 통신 오류): {e}")

# --- 6. 대화 출력 ---
for message in st.session_state.messages:
    role = "Client" if message["role"] == "user" else "Architect"
    avatar = "👤" if message["role"] == "user" else "🛡️"
    with st.chat_message(role, avatar=avatar):
        st.markdown(message['content'], unsafe_allow_html=True)

# --- 7. 입력 및 응답 생성 (★듀얼 RAG 통합 및 속도 제어★) ---

if prompt := st.chat_input("시뮬레이션 변수를 입력하십시오."):
    st.session_state.messages.append({"role": "user", "content": f"<div class='fadein'>{prompt}</div>"})
    with st.chat_message("Client", avatar="👤"):
        st.markdown(f"<div class='fadein'>{prompt}</div>", unsafe_allow_html=True)

    # [★핵심 수정 1: 듀얼 RAG 실행★] LLM 호출 전에 실행
    rag_context = ""
    similar_precedents = [] # 시각화용 저장
    
    # 메뉴 입력이 아닐 경우 RAG 실행
    if not _is_menu_input(prompt):
         with st.spinner("실시간 데이터베이스 분석 중... (Dual RAG: 판례/법령)..."):
            # 1. 법령 검색 (S-RAG)
            if ("statutes" in st.session_state and st.session_state.statutes):
                similar_statutes = find_similar_items(prompt,
                                                     st.session_state.statutes,
                                                     st.session_state.s_embeddings,
                                                     top_k=3, threshold=0.60) # 법령은 임계값 상향
                if similar_statutes:
                    s_texts = [f"[유사도: {c['similarity']:.2f}]\n{c['raw_text']}\n---\n" for c in similar_statutes]
                    rag_context += "\n\n[시스템 참조: 검색된 관련 법령 데이터]\n" + "\n".join(s_texts)

            # 2. 판례 검색 (P-RAG)
            if ("precedents" in st.session_state and st.session_state.precedents):
                similar_precedents = find_similar_items(prompt, 
                                                        st.session_state.precedents, 
                                                        st.session_state.p_embeddings, 
                                                        top_k=5, threshold=0.50)
                if similar_precedents:
                    p_texts = [f"[유사도: {c['similarity']:.2f}]\n{c['raw_text']}\n---\n" for c in similar_precedents]
                    rag_context += "\n\n[시스템 참조: 검색된 유사 판례 데이터]\n" + "\n".join(p_texts)


    # [★핵심 수정 2: 최종 프롬프트 구성★] 사용자 입력 + RAG 컨텍스트 주입
    final_prompt = f"{prompt}\n{rag_context}"

    # 시스템 응답 생성 (API 호출)
    with st.spinner("Architect 시스템 연산 중... 변수 분석 및 시뮬레이션 실행..."):
        try:
            response_stream = st.session_state.chat.send_message(final_prompt, stream=True)
            
            with st.chat_message("Architect", avatar="🛡️"):
                placeholder = st.empty()
                full_response = ""
                
                # [★핵심 수정 3: 스무스 스트리밍 + 속도 제어★]
                word_buffer = ""
                for chunk in response_stream:
                    if getattr(chunk, "text", None):
                        word_buffer += chunk.text
                        
                        # 공백이나 구두점을 만나면 버퍼를 비우고 화면 업데이트
                        if re.search(r'[\s.,!?\n]', chunk.text):
                            full_response += word_buffer
                            word_buffer = ""
                            # 속도 조절을 위한 미세한 지연 (0.01초)
                            time.sleep(0.01) 
                            placeholder.markdown(
                                f"<div class='fadein'>{full_response}▌</div>",
                                unsafe_allow_html=True
                            )
                
                # 마지막 남은 버퍼 처리
                if word_buffer:
                    full_response += word_buffer

                placeholder.markdown(
                    f"<div class='fadein'>{full_response}</div>",
                    unsafe_allow_html=True
                )

            st.session_state.messages.append({"role": "Architect", "content": f"<div class='fadein'>{full_response}</div>"})

            # [★핵심 수정 4: 판례 시각화★] 최종 보고서이고, P-RAG 결과가 있을 경우 표시
            if _is_final_report(full_response) and similar_precedents:
                # (기존 판례 카드 표시 로직 유지)
                q_title = _query_title(prompt)
                st.markdown("**📚 실시간 판례 전문 분석 (P-RAG 결과)**\n\n* 검색 쿼리: `[" + q_title + "]`\n")

                for case_data in similar_precedents[:3]:
                    # 파싱 실행
                    case = _parse_precedent_block(case_data['raw_text'])
                    sim_pct = int(round(case_data["similarity"] * 100))
                    
                    label = f"판례 [{case.get('title','제목 없음')}]"
                    if case.get("court") and case.get("case_no"):
                        label += f" — {case['court']} {case['case_no']}"

                    item_md = (
                        f"* {label}  \n"
                        f"  - 선고: {case.get('date','').strip()} {case.get('court','').strip()} | 유사도: {sim_pct}%  \n"
                        f"  - 판결요지: {case.get('holding','').strip()}  \n"
                        f"  - 전문 일부: \"{case.get('excerpt','').strip()}\""
                    )
                    st.markdown(item_md)

        except Exception as e:
            err = f"시뮬레이션 오류 발생: {e}"
            st.error(err)
            st.session_state.messages.append({"role": "Architect", "content": f"<div class='fadein'>{err}</div>"})
