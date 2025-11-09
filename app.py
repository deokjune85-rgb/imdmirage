# ======================================================
# 🛡️ 베리타스 엔진 7.1 — Hybrid RAG Build (Omega-Infinitum Core)
# ======================================================
import streamlit as st
import google.generativeai as genai
import os
import numpy as np
import re # 정규식 사용

# --- 1. 시스템 설정 (The Vault & Mirage Protocol) ---
st.set_page_config(page_title="베리타스 엔진 7.1", page_icon="🛡️", layout="centered")

# CSS 해킹 (신기루 프로토콜) - 네놈이 넣은 CSS 유지 및 최적화
custom_css = """
<style>
#MainMenu, footer, header, .stDeployButton {visibility:hidden;}

/* --- 글자 스타일 통일 --- */
html, body, div, span, p {
    font-family: 'Noto Sans KR', sans-serif !important;
    /* color: #FFFFFF !important; (다크모드 강제는 가독성을 해침. 주석 처리하여 테마 호환성 확보) */
    font-size: 16px !important; /* 17px은 너무 크다. 16px로 조정 */
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

/* --- 중요 문단 / 헤드라인 컬러 강조 --- */
strong, b {
    /* color: #5AB0FF !important; (포인트 컬러는 유지하되, 테마 자동 조정 권장) */
    font-weight: 700;
}

/* --- 부드러운 텍스트 등장 (속도 개선 0.8s -> 0.5s) --- */
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
st.title("베리타스 엔진 버전 7.1")

st.error("보안 경고: 본 시스템은 격리된 사설 환경(The Vault)에서 작동합니다. 모든 데이터는 기밀로 취급되며 외부로 유출되지 않습니다.")


# --- 3. API 키 및 모델 설정 ---
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
    if not API_KEY:
         raise ValueError("API Key is empty.")
    genai.configure(api_key=API_KEY)
except (KeyError, ValueError) as e:
    st.error(f"시스템 오류: 엔진 연결 실패. (API Key 누락 또는 비어있음): {e}")
    st.stop()

# --- [작전명: 트로이 목마] 게릴라 RAG 엔진 함수 정의 ---
# (네놈이 추가한 RAG 관련 함수들 유지 및 최적화)
EMBEDDING_MODEL_NAME = "models/text-embedding-004"

def embed_text(text, task_type="retrieval_document"):
    # (기존 함수 내용 유지)
    try:
        clean_text = text.replace('\n', ' ').strip()
        if not clean_text:
            return None
        result = genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=clean_text,
            task_type=task_type
        )
        return result['embedding']
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

@st.cache_data
def load_and_embed_precedents(file_path='precedents_data.txt'):
    # (기존 함수 내용 유지 - 견고한 스플릿 포함)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return [], []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return [], []

    chunks = re.split(r'\s*---END OF PRECEDENT---\s*', content)
    precedents = [p.strip() for p in chunks if p and p.strip()]

    embeddings, valid_precedents = [], []
    for p in precedents:
        ebd = embed_text(p, task_type="retrieval_document")
        if ebd:
            embeddings.append(ebd)
            valid_precedents.append(p)

    print(f"[RAG] precedents={len(valid_precedents)}")
    return valid_precedents, embeddings

def _parse_precedent_block(text: str) -> dict:
    # (기존 파싱 함수 내용 유지)
    t = text.strip()
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    title = lines[0][:120] if lines else "제목 없음"
    m = re.search(
        r'\[(?P<court>[^ \[\]]+)\s+(?P<date>\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.)\s*선고\s*(?P<caseno>\d{4}\s*[가-힣]{1,2}\s*\d{3,6})\s*판결\]',
        t
    )
    court = m.group('court') if m else ""
    date  = m.group('date') if m else ""
    caseno = m.group('caseno').replace(" ", "") if (m and m.group('caseno')) else ""

    if not caseno:
        m2 = re.search(r'(?P<caseno>\d{4}\s*[가-힣]{1,2}\s*\d{3,6})', t)
        if m2:
            caseno = m2.group('caseno').replace(" ", "")

    holding = ""
    m2 = re.search(r'【판결요지】(.*?)(【|$)', t, re.S)
    if m2:
        holding = re.sub(r'\s+', ' ', m2.group(1)).strip()
    else:
        m3 = re.search(r'【판시사항】(.*?)(【|$)', t, re.S)
        if m3:
            holding = re.sub(r'\s+', ' ', m3.group(1)).strip()

    if not holding:
        holding = re.sub(r'\s+', ' ', t)[:160].strip()

    excerpt = ""
    for key in ["【전문】", "【이 유】", "【이유】", "【본문】"]:
        pos = t.find(key)
        if pos != -1:
            excerpt = re.sub(r'\s+', ' ', t[pos:pos+300]).strip()
            break
    if not excerpt:
        excerpt = re.sub(r'\s+', ' ', t)[:300].strip()

    if len(holding) > 130: holding = holding[:130].rstrip() + "…"
    if len(excerpt) > 160: excerpt = excerpt[:160].rstrip() + "…"

    return {
        "title": title, "court": court, "date": date,
        "case_no": caseno, "holding": holding, "excerpt": excerpt,
    }

def find_similar_precedents(query_text, precedents, embeddings, top_k=5):
    if not embeddings or not precedents:
        return []

    # task_type 수정: retrieval_query
    q_emb = embed_text(query_text, task_type="retrieval_query")
    if q_emb is None:
        return []

    sims = np.dot(np.array(embeddings), np.array(q_emb))
    idxs = np.argsort(sims)[::-1][:top_k]

    results = []
    for i in idxs:
        sim = float(sims[i])
        if sim < 0.50: # 임계값 상향 조정 0.20 -> 0.50 (정확도 확보)
            continue

        parsed = _parse_precedent_block(precedents[i])
        results.append({
            "similarity": sim,
            "raw_text": precedents[i], # ★중요: LLM 주입을 위해 원본 텍스트 추가★
            **parsed
        })

    return results

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
        # [수정됨] 존재하지 않는 '2.5'가 아니라 '1.5-flash-latest' 사용. 'models/' 접두사 추가.
        st.session_state.model = genai.GenerativeModel("models/gemini-2.5-flash",
                                                    system_instruction=SYSTEM_INSTRUCTION)
        
        # [RAG 초기화]
        with st.spinner("판례 분석 엔진(RAG) 초기화 중... (최초 실행 시)"):
            p, e = load_and_embed_precedents()
            st.session_state.precedents = p
            st.session_state.embeddings = e
            if not p:
                st.warning("⚠️ 판례 데이터(precedents_data.txt) 로드 실패 또는 비어있음. RAG 기능 비활성화.")

    except Exception as e:
        st.error(f"시스템 초기화 실패: {e}")
        st.stop()

# --- 5. 대화 세션 관리 및 자동 시작 ---
# (초기화 로직 강화)
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat" not in st.session_state or not st.session_state.messages:
    if "model" in st.session_state:
        try:
            if "chat" not in st.session_state:
                st.session_state.chat = st.session_state.model.start_chat(history=[])

            if not st.session_state.messages:
                # 초기화 명령 강화
                initial_prompt = "긴급 명령: EPE 활성화. 즉시 <KnowledgeBase>의 'Phase 0: 도메인 선택 프로토콜'을 실행하고 메뉴를 출력하라. 다른 설명이나 확인은 생략한다."
                response = st.session_state.chat.send_message(initial_prompt)
                if response and response.text:
                     # 시각 효과(fadein) 적용하여 저장
                     st.session_state.messages.append({"role": "Architect", "content": f"<div class='fadein'>{response.text}</div>"})
                else:
                     st.error("시스템 코어 응답 실패 (응답 없음).")
        except Exception as e:
            st.error(f"시스템 초기화 실패 (API 통신 오류): {e}")


# --- 6. 대화 출력 (기존 로직 유지) ---
for message in st.session_state.messages:
    role = "Client" if message["role"] == "user" else "Architect"
    avatar = "👤" if message["role"] == "user" else "🛡️"
    with st.chat_message(role, avatar=avatar):
        # 이미 fadein이 적용된 HTML이므로 그대로 출력
        st.markdown(message['content'], unsafe_allow_html=True)

# --- 7. 입력 및 응답 생성 (★핵심 수정: 하이브리드 RAG★) ---

# 유틸리티 함수 정의 (기존 유지 및 개선)
def _is_menu_input(s: str) -> bool:
    if not s: return False
    return bool(re.fullmatch(r'\d+|[1-9]-\d+', s.strip())) # 계층형 메뉴 대응 수정

def _is_final_report(txt: str) -> bool:
    if not txt: return False
    t = txt.replace(" ", "")
    hits = 0
    # 키워드 수정: 실제 보고서 키워드 반영
    for key in ["전략브리핑보고서", "리스크시뮬레이션분석", "권장다음단계", "면책조항"]:
        if key in t: hits += 1
    return (hits >= 2) and (len(t) > 500) # 길이 기준 완화 800 -> 500

def _query_title(prompt_text: str) -> str:
    # (기존 함수 내용 유지)
    if not prompt_text: return ""
    m = re.search(r'\[([^\]]+)\]', prompt_text)
    if m: return m.group(1).strip()
    first = prompt_text.strip().splitlines()[0].strip()
    return (first[:77] + "…") if len(first) > 80 else first


if prompt := st.chat_input("시뮬레이션 변수를 입력하십시오."):
    # 사용자 입력 표시 시 fadein 적용
    st.session_state.messages.append({"role": "user", "content": f"<div class='fadein'>{prompt}</div>"})
    with st.chat_message("Client", avatar="👤"):
        st.markdown(f"<div class='fadein'>{prompt}</div>", unsafe_allow_html=True)

    # [★핵심 수정 1: RAG 실행 시점 이동★] LLM 호출 전에 RAG 실행
    rag_context = ""
    similar_cases = [] # 카드 표시를 위해 저장
    
    # 메뉴 입력이 아니고, 데이터가 로드된 경우에만 RAG 실행
    if not _is_menu_input(prompt) and ("precedents" in st.session_state and st.session_state.precedents):
         with st.spinner("실시간 판례 데이터베이스 분석 중... 유사 사례 검색(RAG)..."):
            similar_cases = find_similar_precedents(prompt, 
                                                    st.session_state.precedents, 
                                                    st.session_state.embeddings, 
                                                    top_k=5)
            if similar_cases:
                # LLM 주입용 컨텍스트 생성 (원본 텍스트 사용)
                rag_texts = [f"[유사도: {c['similarity']:.2f}]\n{c['raw_text']}\n---\n" for c in similar_cases]
                rag_context = "\n\n[시스템 참조: 검색된 유사 판례 데이터]\n" + "\n".join(rag_texts)

    # [★핵심 수정 2: 최종 프롬프트 구성★] 사용자 입력 + RAG 컨텍스트 주입
    final_prompt = f"{prompt}\n{rag_context}"

    # 시스템 응답 생성 (API 호출)
    with st.spinner("Architect 시스템 연산 중... 변수 분석 및 시뮬레이션 실행..."):
        try:
            # [★핵심 수정 3: final_prompt 사용★] 증강된 프롬프트를 LLM에 전송
            response_stream = st.session_state.chat.send_message(final_prompt, stream=True)
            
            with st.chat_message("Architect", avatar="🛡️"):
                placeholder = st.empty()
                full_response = ""
                for chunk in response_stream:
                    if getattr(chunk, "text", None):
                        full_response += chunk.text
                        # 스트리밍 출력 시 fadein 적용
                        placeholder.markdown(
                            f"<div class='fadein'>{full_response}▌</div>",
                            unsafe_allow_html=True
                        )
                placeholder.markdown(
                    f"<div class='fadein'>{full_response}</div>",
                    unsafe_allow_html=True
                )

            # 스트림 폴백 로직 (단순화)
            if not full_response.strip():
                 pass

            # 최종 응답 저장 시 fadein 적용
            st.session_state.messages.append({"role": "Architect", "content": f"<div class='fadein'>{full_response}</div>"})

            # [★핵심 수정 4: 판례 카드 표시★] 이미 계산된 similar_cases 사용
            # 최종 보고서이고, RAG 결과가 있을 경우에만 표시 (네가 원하던 UI)
            if _is_final_report(full_response) and similar_cases:
                # 헤더 출력
                q_title = _query_title(prompt)
                st.markdown("**📚 실시간 판례 전문 분석 (RAG 결과)**\n\n* 검색 쿼리: `[" + q_title + "]`\n")

                # 상위 3건만 카드형 요약으로 출력
                for case in similar_cases[:3]:
                    sim_pct = int(round(case["similarity"] * 100))
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
            elif _is_final_report(full_response) and not _is_menu_input(prompt) and not similar_cases:
                 # 보고서는 나왔지만 RAG 결과가 없을 경우 (임계값 미달 등)
                 st.info("ℹ️ 분석과 관련된 유사 판례가 데이터베이스에서 검색되지 않았습니다. (임계값 0.50)")


        except Exception as e:
            err = f"시뮬레이션 오류 발생: {e}"
            st.error(err)
            # 오류 메시지 저장 시 fadein 적용
            st.session_state.messages.append({"role": "Architect", "content": f"<div class='fadein'>{err}</div>"})
