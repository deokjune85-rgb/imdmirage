# ======================================================
# 🛡️ 베리타스 엔진 7.1 — Fine-Tune Build (윤진 커스텀 완성본)
# ======================================================
import streamlit as st
import google.generativeai as genai
import os
import numpy as np

# --- 1. 시스템 설정 (The Vault & Mirage Protocol) ---
st.set_page_config(page_title="베리타스 엔진 7.1", page_icon="🛡️", layout="centered")

# CSS 해킹 (신기루 프로토콜)
custom_css = """
<style>
#MainMenu, footer, header, .stDeployButton {visibility:hidden;}

/* --- 글자 스타일 통일 --- */
html, body, div, span, p {
    font-family: 'Noto Sans KR', sans-serif !important;
    color: #FFFFFF !important;
    font-size: 17px !important;
    line-height: 1.7 !important;
}

/* --- 타이틀 위치 조정 (여백 최소화) --- */
h1 {
    text-align: left !important;
    font-weight: 900 !important;
    font-size: 36px !important;
    margin-top: 10px !important;
    margin-bottom: 15px !important;
    color: #FFFFFF !important;
}

/* --- 중요 문단 / 헤드라인 컬러 강조 --- */
strong, b {
    color: #5AB0FF !important; /* 진파랑 포인트 */
}

/* --- 부드러운 텍스트 등장 (제미나이형 시각 효과) --- */
.fadein {
    animation: fadeInText 0.8s ease-in-out forwards;
    opacity: 0;
}
@keyframes fadeInText {
    from {opacity: 0; transform: translateY(3px);}
    to {opacity: 1; transform: translateY(0);}
}

/* --- 판례/결과 출력 시 텍스트 통일 --- */
[data-testid="stChatMessageContent"] {
    font-size: 17px !important;
    color: #FFFFFF !important;
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
except KeyError:
    st.error("시스템 오류: 엔진 연결 실패. (API Key 누락)")
    st.stop()

genai.configure(api_key=API_KEY)

# --- [작전명: 트로이 목마] 게릴라 RAG 엔진 함수 정의 ---
EMBEDDING_MODEL_NAME = "models/text-embedding-004"

def embed_text(text, task_type="RETRIEVAL_DOCUMENT"):
    try:
        clean_text = text.replace('\n', ' ').strip()
        if not clean_text:
            return None
        # task_type은 "RETRIEVAL_DOCUMENT" / "RETRIEVAL_QUERY" 만 사용
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
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return [], []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return [], []

    # 견고한 스플릿: 마커 라인에 공백/개행 있어도 분할
    import re
    chunks = re.split(r'\s*---END OF PRECEDENT---\s*', content)
    precedents = [p.strip() for p in chunks if p and p.strip()]

    embeddings, valid_precedents = [], []
    for p in precedents:
        ebd = embed_text(p, task_type="RETRIEVAL_DOCUMENT")
        if ebd:
            embeddings.append(ebd)
            valid_precedents.append(p)

    print(f"[RAG] precedents={len(valid_precedents)}")
    return valid_precedents, embeddings


def _parse_precedent_block(text: str) -> dict:
    """프리텍스트 판례 블록에서 제목/선고/요지/발췌를 최대한 뽑아낸다(룰베이스)."""
    import re
    t = text.strip()

    # 제목(첫 줄 또는 대법원/고등법원 헤더)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    title = lines[0][:120] if lines else "제목 없음"

    # [대법원 2024. 1. 18. 선고 ... 판결] 패턴에서 법원/선고일자 추출
    m = re.search(r'\[(?P<court>[^ \[\]]+)\s+(?P<date>\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.)\s*선고.*?판결\]', t)
    court = m.group('court') if m else ""
    date  = m.group('date') if m else ""

    # 【판결요지】 또는 【판시사항】 일부 추출
    holding = ""
    m2 = re.search(r'【판결요지】(.*?)(【|$)', t, re.S)
    if m2:
        holding = re.sub(r'\s+', ' ', m2.group(1)).strip()
    else:
        m3 = re.search(r'【판시사항】(.*?)(【|$)', t, re.S)
        if m3:
            holding = re.sub(r'\s+', ' ', m3.group(1)).strip()

    if not holding:
        # 없으면 본문 초반 160자 정도로 대체
        holding = re.sub(r'\s+', ' ', t)[:160].strip()

    # 전문 일부(전문/이유/본문 근처에서 120~160자)
    excerpt = ""
    for key in ["【전문】", "【이 유】", "【이유】", "【본문】"]:
        pos = t.find(key)
        if pos != -1:
            excerpt = re.sub(r'\s+', ' ', t[pos:pos+300]).strip()
            break
    if not excerpt:
        excerpt = re.sub(r'\s+', ' ', t)[:300].strip()

    # 좀 줄여주기
    if len(holding) > 130: holding = holding[:130].rstrip() + "…"
    if len(excerpt) > 160: excerpt = excerpt[:160].rstrip() + "…"

    return {
        "title": title,
        "court": court,
        "date":  date,
        "holding": holding,
        "excerpt": excerpt,
    }


def find_similar_precedents(query_text, precedents, embeddings, top_k=3):
    """
    기존: 커다란 전문 문자열을 그대로 반환
    변경: 깔끔한 요약카드용 dict 목록 반환
    """
    if not embeddings or not precedents:
        return []

    q_emb = embed_text(query_text, task_type="search_query")
    if q_emb is None:
        return []

    embeddings_np = np.array(embeddings)
    q_np = np.array(q_emb)
    sims = np.dot(embeddings_np, q_np)

    # 상위 K개
    idxs = np.argsort(sims)[::-1][:top_k]

    results = []
    for i in idxs:
        sim = float(sims[i])
        # 임계값 너무 높으면 안 나오는 문제 → 살짝 완화(0.20)
        if sim < 0.20:
            continue

        parsed = _parse_precedent_block(precedents[i])
        results.append({
            "similarity": sim,  # 0~1
            **parsed
        })

    return results





# --- 4. 시스템 프라임 유전자 (Prime Genome) ---
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_INSTRUCTION = f.read()

if "model" not in st.session_state:
    st.session_state.model = genai.GenerativeModel("gemini-2.5-flash",
                                                   system_instruction=SYSTEM_INSTRUCTION)

# --- 5. 대화 세션 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat" not in st.session_state:
    st.session_state.chat = st.session_state.model.start_chat(history=[])
    initial_prompt = "시스템 가동. '동적 라우팅 프로토콜'을 실행하여 Phase 0를 시작하라."
    try:
        response = st.session_state.chat.send_message(initial_prompt)
        st.session_state.messages.append({"role": "Architect", "content": f"<div class='fadein'>{response.text}</div>"})
    except Exception as e:
        st.error(f"시스템 초기화 실패: {e}")


# --- 6. 대화 출력 ---
for message in st.session_state.messages:
    role = "Client" if message["role"] == "user" else "Architect"
    avatar = "👤" if message["role"] == "user" else "🛡️"
    with st.chat_message(role, avatar=avatar):
        st.markdown(f"<div class='fadein'>{message['content']}</div>", unsafe_allow_html=True)

# --- 7. 입력 및 마지막 Phase에서만 판례 호출 (브리핑 보고서 트리거 버전) ---
import re

def _is_menu_input(s: str) -> bool:
    if not s:
        return False
    s = s.strip()
    # 숫자만, 또는 2-숫자 형태만 (메뉴 선택)
    return bool(re.fullmatch(r'\d+|2-\d+', s))

def _is_final_report(txt: str) -> bool:
    if not txt:
        return False
    t = txt.replace(" ", "")
    # '최종 보고서' 포맷의 핵심 표지어가 최소 2개 이상 존재 + 길이 기준
    hits = 0
    for key in ["유사수신/사기전략브리핑보고서",
                "리스크시뮬레이션분석",
                "권장다음단계",
                "면책조항",
                "최종보고서",
                "브리핑보고서"]:
        if key in t:
            hits += 1
    return (hits >= 2) and (len(t) > 800)

if prompt := st.chat_input("시뮬레이션 변수를 입력하십시오."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("Client", avatar="👤"):
        st.markdown(f"<div class='fadein'>{prompt}</div>", unsafe_allow_html=True)

    with st.spinner("Architect 시스템 연산 중..."):
        try:
            response_stream = st.session_state.chat.send_message(prompt, stream=True)
            with st.chat_message("Architect", avatar="🛡️"):
                placeholder = st.empty()
                full_response = ""
                for chunk in response_stream:
                    # 일부 응답 조각이 비어있는 경우가 있어 가드
                    if not getattr(chunk, "text", None):
                        continue
                    full_response += chunk.text
                    placeholder.markdown(
                        f"<div class='fadein'>{full_response}▌</div>",
                        unsafe_allow_html=True
                    )
                placeholder.markdown(
                    f"<div class='fadein'>{full_response}</div>",
                    unsafe_allow_html=True
                )

            # 스트림이 비어 있으면 non-stream 폴백
            if not full_response.strip():
                non_stream = st.session_state.chat.send_message(prompt)
                txt = getattr(non_stream, "text", None)
                if txt:
                    full_response = txt
                    with st.chat_message("Architect", avatar="🛡️"):
                        st.markdown(f"<div class='fadein'>{full_response}</div>", unsafe_allow_html=True)

            st.session_state.messages.append({"role": "Architect", "content": full_response})

            # 🔒 여기서 '최종 보고서'일 때만 판례 붙임 (메뉴 입력/중간 단계에서는 절대 안 붙임)
            if _is_final_report(full_response) and not _is_menu_input(prompt):
                precedents, embeddings = load_and_embed_precedents()
                if not precedents or not embeddings:
                    st.warning("⚠️ 판례 탄약고가 비었거나 로드 실패. 'precedents_data.txt' 위치/형식 확인.")
                else:
                    similar_cases = find_similar_precedents(prompt, precedents, embeddings, top_k=5)
                    if similar_cases:
                        st.markdown("<br><b>📚 실시간 판례 전문 분석</b><br>", unsafe_allow_html=True)
                        # 과도한 줄바꿈 방지
                            if similar_cases:
        # 헤더 + 검색 쿼리
        st.markdown("**📚 실시간 판례 전문 분석**\n\n* 검색 쿼리: `" + prompt + "`\n")

        # 상위 3건만 카드형 요약으로 출력
        for case in similar_cases[:3]:
            sim_pct = int(round(case["similarity"] * 100))
            item_md = (
                f"* 판례 [{case.get('title','제목 없음')}]  \n"
                f"  - 선고: {case.get('date','').strip()} {case.get('court','').strip()} | 유사도: {sim_pct}%  \n"
                f"  - 판결요지: {case.get('holding','').strip()}  \n"
                f"  - 전문 일부: \"{case.get('excerpt','').strip()}\""
            )
            st.markdown(item_md)

                    else:
                        st.info("ℹ️ 최종 보고서 기준으로 매칭된 유사 판례가 없습니다. (임계값 0.20)")

        except Exception as e:
            err = f"시뮬레이션 오류 발생: {e}"
            st.error(err)
            st.session_state.messages.append({"role": "Architect", "content": err})
