# ======================================================
# 🛡️ 베리타스 엔진 7.3 — Dual RAG + Relay Mechanism (Omega-Infinitum Core)
# ======================================================
import streamlit as st
import google.generativeai as genai
import os
import numpy as np
import re
import time # 속도 조절을 위해 필요함

# --- 1. 시스템 설정 (The Vault & Mirage Protocol) ---
# 테마 설정: 시스템 기본값 사용 (흰 바탕/검은 글씨 또는 다크 모드 자동 호환)
st.set_page_config(page_title="베리타스 엔진 7.3", page_icon="🛡️", layout="centered")

# CSS 해킹 (신기루 프로토콜) - [★수정됨: 색상 강제 제거 및 최적화]
custom_css = """
<style>
#MainMenu, footer, header, .stDeployButton {visibility:hidden;}

/* --- 글자 스타일 통일 (색상 강제 제거) --- */
html, body, div, span, p {
    font-family: 'Noto Sans KR', sans-serif !important;
    /* color: #FFFFFF !important; <-- 이전 코드의 흰 글씨 문제 원인 제거. */
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

strong, b {
    font-weight: 700;
}

/* --- 부드러운 텍스트 등장 (0.5s) --- */
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
st.title("베리타스 엔진 버전 7.3")
# 라이트 모드 가독성을 위해 warning 사용
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

# --- [RAG 엔진 함수 정의] (기존 내용 유지) ---
EMBEDDING_MODEL_NAME = "models/text-embedding-004"

def embed_text(text, task_type="retrieval_document"):
    try:
        clean_text = text.replace('\n', ' ').strip()
        if not clean_text: return None
        result = genai.embed_content(model=EMBEDDING_MODEL_NAME, content=clean_text, task_type=task_type)
        return result['embedding']
    except Exception as e:
        print(f"Embedding error: {e}"); return None

@st.cache_data
def load_and_embed_data(file_path, separator_regex):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}"); return [], []
    try:
        with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}"); return [], []
    if not content.strip(): return [], []
    chunks = re.split(separator_regex, content)
    data_items = [p.strip() for p in chunks if p and p.strip()]
    embeddings, valid_items = [], []
    for item in data_items:
        ebd = embed_text(item, task_type="retrieval_document")
        if ebd:
            embeddings.append(ebd); valid_items.append(item)
    print(f"[RAG] Loaded {len(valid_items)} items from {file_path}.")
    return valid_items, embeddings

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
    # HTML 태그 제거 후 분석 (릴레이 메커니즘 대응 강화)
    t = re.sub('<[^<]+?>', '', txt).replace(" ", "")
    hits = 0
    # 보고서 식별 키워드 강화
    for key in ["전략브리핑보고서", "리스크시뮬레이션분석", "권장다음단계", "면책조항", "시뮬레이션완료"]:
        if key in t: hits += 1
    return (hits >= 2) and (len(t) > 500)

def _query_title(prompt_text: str) -> str:
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
        st.session_state.model = genai.GenerativeModel("models/gemini-1.5-flash-latest",
                                                    system_instruction=SYSTEM_INSTRUCTION)
        
        # 듀얼 RAG 초기화
        with st.spinner("분석 엔진(Dual RAG) 초기화 중... (최초 실행 시)"):
            # 1. 판례 데이터 로드 (P-RAG)
            p_data, p_emb = load_and_embed_data('precedents_data.txt', r'\s*---END OF PRECEDENT---\s*')
            st.session_state.precedents = p_data
            st.session_state.p_embeddings = p_emb

            # 2. 법령 데이터 로드 (S-RAG)
            s_data, s_emb = load_and_embed_data('statutes_data.txt', r'\s*---END OF STATUTE---\s*')
            st.session_state.statutes = s_data
            st.session_state.s_embeddings = s_emb

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
        except Exception as e:
            st.error(f"시스템 초기화 실패 (API 통신 오류): {e}")

# --- 6. 대화 출력 ---
for message in st.session_state.messages:
    role = "Client" if message["role"] == "user" else "Architect"
    avatar = "👤" if message["role"] == "user" else "🛡️"
    with st.chat_message(role, avatar=avatar):
        st.markdown(message['content'], unsafe_allow_html=True)

# --- 7. 입력 및 응답 생성 (★핵심 수정: 릴레이 메커니즘 탑재★) ---

# [★신설★] 스트리밍 출력 및 저장 함수 (속도 제어 포함)
def stream_and_store_response(chat_session, prompt_to_send, spinner_text="Architect 시스템 연산 중..."):
    full_response = ""
    with st.spinner(spinner_text):
        try:
            response_stream = chat_session.send_message(prompt_to_send, stream=True)
            
            with st.chat_message("Architect", avatar="🛡️"):
                placeholder = st.empty()
                word_buffer = ""
                try:
                    for chunk in response_stream:
                        if getattr(chunk, "text", None):
                            word_buffer += chunk.text
                            # 스무스 스트리밍 + 속도 제어 (0.01초 지연으로 멀미 방지)
                            if re.search(r'[\s.,!?\n]', chunk.text):
                                full_response += word_buffer
                                word_buffer = ""
                                time.sleep(0.01) 
                                placeholder.markdown(
                                    f"<div class='fadein'>{full_response}▌</div>",
                                    unsafe_allow_html=True
                                )
                except Exception as stream_error:
                     # 스트리밍 중단 시 (과부하/타임아웃) 오류 표시
                     full_response += f"\n\n[⚠️ 시스템 과부하 감지: 응답 생성 중단됨. {stream_error} ⚠️]"

                if word_buffer:
                    full_response += word_buffer
                
                placeholder.markdown(
                    f"<div class='fadein'>{full_response}</div>",
                    unsafe_allow_html=True
                )
            
            # 메시지 저장 (HTML 포함)
            st.session_state.messages.append({"role": "Architect", "content": f"<div class='fadein'>{full_response}</div>"})
            return full_response

        except Exception as e:
            err = f"시뮬레이션 오류 발생 (API 호출 실패): {e}"
            st.error(err)
            st.session_state.messages.append({"role": "Architect", "content": f"<div class='fadein'>{err}</div>"})
            return err

# 메인 입력 루프
if prompt := st.chat_input("시뮬레이션 변수를 입력하십시오."):
    st.session_state.messages.append({"role": "user", "content": f"<div class='fadein'>{prompt}</div>"})
    with st.chat_message("Client", avatar="👤"):
        st.markdown(f"<div class='fadein'>{prompt}</div>", unsafe_allow_html=True)

    # [★핵심 수정 1: Phase 2 감지★]
    is_phase2_data = False
    if st.session_state.messages:
        # 마지막 Architect 메시지 찾기 (HTML 제거 후 분석)
        last_architect_msg = ""
        for msg in reversed(st.session_state.messages):
            if msg['role'] == 'Architect':
                # HTML 태그 제거 후 분석
                last_architect_msg = re.sub('<[^<]+?>', '', msg['content'])
                break
        
        # 이전 메시지가 Phase 2 데이터 요청이었는지 확인 (키워드 기반 감지)
        # (Phase 2 요청 문구는 system_prompt.txt에 정의된 내용을 기반으로 함)
        if "Phase 2:" in last_architect_msg and ("데이터를 지금 시스템에 입력하십시오" in last_architect_msg or "엔진'을 가동하여" in last_architect_msg):
            is_phase2_data = True

    # [★핵심 수정 2: 듀얼 RAG 실행★]
    rag_context = ""
    similar_precedents = []
    
    if not _is_menu_input(prompt):
         with st.spinner("실시간 데이터베이스 분석 중... (Dual RAG: 판례/법령)..."):
            # 1. 법령 검색 (S-RAG)
            if ("statutes" in st.session_state and st.session_state.statutes):
                similar_statutes = find_similar_items(prompt, st.session_state.statutes, st.session_state.s_embeddings, top_k=3, threshold=0.60)
                if similar_statutes:
                    s_texts = [f"[유사도: {c['similarity']:.2f}]\n{c['raw_text']}\n---\n" for c in similar_statutes]
                    rag_context += "\n\n[시스템 참조: 검색된 관련 법령 데이터]\n" + "\n".join(s_texts)

            # 2. 판례 검색 (P-RAG)
            if ("precedents" in st.session_state and st.session_state.precedents):
                similar_precedents = find_similar_items(prompt, st.session_state.precedents, st.session_state.p_embeddings, top_k=5, threshold=0.50)
                if similar_precedents:
                    p_texts = [f"[유사도: {c['similarity']:.2f}]\n{c['raw_text']}\n---\n" for c in similar_precedents]
                    rag_context += "\n\n[시스템 참조: 검색된 유사 판례 데이터]\n" + "\n".join(p_texts)

    # 최종 프롬프트 구성
    final_prompt = f"{prompt}\n{rag_context}"

    # 시스템 응답 생성 (Phase 1 또는 Phase 2 분석)
    current_response = stream_and_store_response(st.session_state.chat, final_prompt)

    # [★핵심 수정 3: 릴레이 메커니즘★]
    # Phase 2 데이터가 입력되었고, 방금 생성된 응답이 최종 보고서가 아니라면 (즉, 분석 결과만 출력하고 멈췄다면)
    
    # 응답 클린징 (HTML 제거)
    clean_response = re.sub('<[^<]+?>', '', current_response)
    
    if is_phase2_data and not _is_final_report(clean_response):
        # 강제로 Phase 3 실행 명령 (Relay Prompt)
        # 시스템에게 명확하게 다음 단계를 지시한다.
        relay_prompt = "[시스템 명령]: Phase 2 분석 결과 확인 완료. 즉시 이어서 Phase 3(최종 보고서 생성)를 실행하라. 방금 분석한 내용과 RAG 데이터를 바탕으로 보고서 전체를 완성하라. 다른 설명이나 확인은 생략한다."
        
        # 릴레이 프롬프트 실행 (Phase 3 보고서 생성)
        current_response = stream_and_store_response(st.session_state.chat, relay_prompt, spinner_text="최종 보고서 생성 중 (Phase 3 Relay)...")
        # 릴레이 후 응답 다시 클린징
        clean_response = re.sub('<[^<]+?>', '', current_response)


    # [★핵심 수정 4: 판례 시각화★]
    # 최종 응답(릴레이 포함)이 보고서이고, P-RAG 결과가 있을 경우 표시
    if _is_final_report(clean_response) and similar_precedents:
        q_title = _query_title(prompt)
        st.markdown("**📚 실시간 판례 전문 분석 (P-RAG 결과)**\n\n* 검색 쿼리: `[" + q_title + "]`\n")

        for case_data in similar_precedents[:3]:
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
