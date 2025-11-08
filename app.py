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

def embed_text(text, task_type="retrieval_document"):
    try:
        clean_text = text.replace('\n', ' ').strip()
        if not clean_text:
            return None
        result = genai.embed_content(model=EMBEDDING_MODEL_NAME, content=clean_text, task_type=task_type)
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
    precedents = [p.strip() for p in content.split('---END OF PRECEDENT---') if p.strip()]
    embeddings, valid_precedents = [], []
    for p in precedents:
        ebd = embed_text(p)
        if ebd:
            embeddings.append(ebd)
            valid_precedents.append(p)
    print(f"Successfully loaded and embedded {len(valid_precedents)} precedents.")
    return valid_precedents, embeddings

def find_similar_precedents(query_text, precedents, embeddings, top_k=5):
    if not embeddings or not precedents:
        return []

    # 쿼리 임베딩
    query_embedding = embed_text(query_text, task_type="search_query")
    if query_embedding is None:
        return []

    embeddings_np = np.array(embeddings)
    q_np = np.array(query_embedding)

    # text-embedding-004는 보통 단위 정규화되어 있어 내적 ≈ 코사인 유사도
    sims = np.dot(embeddings_np, q_np)

    # 상위 K개
    order = np.argsort(sims)[::-1][:top_k]

    results = []
    for idx in order:
        # 🔽 임계값 완화: 0.20 (너무 깐깐하면 아무 것도 안 뜸)
        if sims[idx] >= 0.20:
            # 과도한 줄바꿈만 최소화
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
if prompt := st.chat_input("시뮬레이션 변수를 입력하십시오."):
    st.session_state["did_precedent"] = False  # 매 턴 리셋

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

            if not full_response.strip():
                non_stream_resp = st.session_state.chat.send_message(prompt)
                try:
                    text_part = getattr(non_stream_resp, "text", None)
                    if text_part:
                        full_response = text_part
                except Exception:
                    pass
                if full_response.strip():
                    with st.chat_message("Architect", avatar="🛡️"):
                        st.markdown(f"<div class='fadein'>{full_response}</div>", unsafe_allow_html=True)

            st.session_state.messages.append({"role": "Architect", "content": full_response})

            # ✅ 매 턴 한 번은 강제 판례 시도
            if st.session_state.get("did_precedent") is False:
                precedents, embeddings = load_and_embed_precedents()

                # 🔽 추가: 탄약고 비었을 때 즉시 안내(왜 안 나오는지 바로 보이게)
                if not precedents or not embeddings:
                    st.warning("⚠️ 판례 탄약고가 비었거나 로드에 실패했습니다. 'precedents_data.txt' 파일을 앱 실행 디렉토리에 두세요.")
                else:
                    similar_cases = find_similar_precedents(prompt, precedents, embeddings)
                    if similar_cases:
                        st.markdown("<br><b>📚 실시간 판례 전문 분석</b><br>", unsafe_allow_html=True)
                        for case in similar_cases:
                            cleaned = case.replace("\n\n\n", "\n\n")
                            st.markdown(f"<div class='fadein'>{cleaned}</div>", unsafe_allow_html=True)

                st.session_state["did_precedent"] = True

        except Exception as e:
            err = f"시뮬레이션 오류 발생: {e}"
            st.error(err)
            st.session_state.messages.append({"role": "Architect", "content": err})

