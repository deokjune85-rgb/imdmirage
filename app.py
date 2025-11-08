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


def find_similar_precedents(query_text, precedents, embeddings, top_k=5):
    if not embeddings or not precedents:
        return []

    q_emb = embed_text(query_text, task_type="RETRIEVAL_QUERY")
    if q_emb is None:
        return []

    import numpy as np
    M = np.array(embeddings, dtype=float)      # (N, D)
    q = np.array(q_emb, dtype=float)           # (D,)

    # 코사인 유사도
    M_norm = np.linalg.norm(M, axis=1) + 1e-12
    q_norm = np.linalg.norm(q) + 1e-12
    sims = (M @ q) / (M_norm * q_norm)

    order = np.argsort(sims)[::-1][:top_k]

    results = []
    for idx in order:
        if sims[idx] >= 0.20:  # 완화
            snippet = precedents[idx].replace("\r", "").replace("\n\n\n", "\n\n")
            results.append(
                f"[유사 판례 발견 (유사도: {sims[idx]:.2f})]\n{snippet}\n---\n"
            )
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
# --- 7. 입력 및 마지막 Phase에서만 판례 호출 (브리핑 보고서 트리거 버전) ---
if prompt := st.chat_input("시뮬레이션 변수를 입력하십시오."):
    st.session_state["did_precedent"] = False  # 🔹(추가) 매 턴 리셋

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

            # 스트림이 비었으면 non-stream 폴백
            if not full_response.strip():
                non_stream = st.session_state.chat.send_message(prompt)
                try:
                    txt = getattr(non_stream, "text", None)
                    if txt:
                        full_response = txt
                except Exception:
                    pass
                if full_response.strip():
                    with st.chat_message("Architect", avatar="🛡️"):
                        st.markdown(f"<div class='fadein'>{full_response}</div>", unsafe_allow_html=True)

            st.session_state.messages.append({"role": "Architect", "content": full_response})

            # 🔹(추가) 디버그: 탄약고 카운트 찍기
            precedents, embeddings = load_and_embed_precedents()
            st.session_state["__dbg_counts__"] = (len(precedents), len(embeddings))

            # 🔹(추가) 강제 1회 판례 부착 (면책이든 뭐든, 매 턴 한 번은 붙임)
            if st.session_state.get("did_precedent") is False:
                if not precedents or not embeddings:
                    st.warning("⚠️ 판례 탄약고가 비었거나 로드 실패. 'precedents_data.txt' 위치/형식 확인.")
                else:
                    similar_cases = find_similar_precedents(prompt, precedents, embeddings)
                    if similar_cases:
                        st.markdown("<br><b>📚 실시간 판례 전문 분석</b><br>", unsafe_allow_html=True)
                        for case in similar_cases:
                            cleaned = case.replace("\n\n\n", "\n\n")
                            st.markdown(f"<div class='fadein'>{cleaned}</div>", unsafe_allow_html=True)
                    else:
                        st.info("ℹ️ 유사 판례가 0건입니다. (임계값 0.20) — 쿼리를 더 구체적으로 입력해 보세요.")
                st.session_state["did_precedent"] = True

            # 🔹(추가) 최소 디버그 패널 (보이기만 함 / UI 불변)
            try:
                c_pre, c_emb = st.session_state.get("__dbg_counts__", (0,0))
                print(f"[RAG] precedents={c_pre}, embeddings={c_emb}")
            except Exception:
                pass

        except Exception as e:
            err = f"시뮬레이션 오류 발생: {e}"
            st.error(err)
            st.session_state.messages.append({"role": "Architect", "content": err})

