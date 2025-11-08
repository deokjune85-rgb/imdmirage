import streamlit as st
import google.generativeai as genai
import os
import numpy as np  # RAG 엔진을 위한 벡터 연산 라이브러리


# --- 1. 시스템 설정 (The Vault & Mirage Protocol) ---
st.set_page_config(page_title="베리타스 엔진 7.1", page_icon="🛡️", layout="centered")

# CSS 스타일 (그라데이션 + 색상 통일)
st.markdown("""
<style>
html, body, div, p, span {
    font-family: 'Noto Sans KR', sans-serif !important;
    font-size: 17px !important;
    color: #000000 !important;
    line-height: 1.6 !important;
}

h1 {
    text-align: left !important;
    font-weight: 900 !important;
    font-size: 34px !important;
    color: #000000 !important;
    margin-top: 10px !important;
}

strong, b {
    color: #003399 !important; /* 진파랑 강조 */
}

/* 스트리밍 시 자연스러운 Fade-in */
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(6px);}
    to {opacity: 1; transform: translateY(0);}
}
.lineblock {
    animation: fadeIn 0.6s ease-in-out forwards;
    white-space: pre-wrap;
}

/* Streamlit 기본 요소 숨기기 */
#MainMenu, footer, header, .stDeployButton {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

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


# --- [작전명: 트로이 목마] 게릴라 RAG 엔진 ---
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
        return [], []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return [], []

    precedents = content.split('---END OF PRECEDENT---')
    precedents = [p.strip() for p in precedents if p.strip()]

    embeddings, valid_precedents = [], []
    for precedent in precedents:
        embedding = embed_text(precedent)
        if embedding:
            embeddings.append(embedding)
            valid_precedents.append(precedent)
    return valid_precedents, embeddings


def find_similar_precedents(query_text, precedents, embeddings, top_k=3):
    if not embeddings or not precedents:
        return []
    query_embedding = embed_text(query_text, task_type="search_query")
    if query_embedding is None:
        return []

    embeddings_np = np.array(embeddings)
    query_embedding_np = np.array(query_embedding)
    similarities = np.dot(embeddings_np, query_embedding_np)

    top_k_indices = np.argsort(similarities)[::-1][:top_k]
    results = []
    for idx in top_k_indices:
        if similarities[idx] > 0.6:
            results.append(f"[유사 판례 발견 (유사도: {similarities[idx]:.2f})]\n{precedents[idx]}\n---\n")
    return results


# --- 4. 시스템 프라임 지침 불러오기 ---
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_INSTRUCTION = f.read()

if "model" not in st.session_state:
    st.session_state.model = genai.GenerativeModel(
        "gemini-2.5-flash",
        system_instruction=SYSTEM_INSTRUCTION
    )


# --- 5. 세션 초기화 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat" not in st.session_state:
    st.session_state.chat = st.session_state.model.start_chat(history=[])

    # 초기 Phase 실행
    initial_prompt = "시스템 가동. '동적 라우팅 프로토콜'을 실행하여 Phase 0를 시작하라."
    try:
        response = st.session_state.chat.send_message(initial_prompt)
        st.session_state.messages.append({"role": "Architect", "content": response.text})
    except Exception as e:
        st.error(f"시스템 초기화 실패: {e}")


# --- 6. 메시지 출력 ---
for message in st.session_state.messages:
    role_name = "Client" if message["role"] == "user" else "Architect"
    avatar = "👤" if message["role"] == "user" else "🛡️"
    with st.chat_message(role_name, avatar=avatar):
        st.markdown(f"<div class='lineblock'>{message['content']}</div>", unsafe_allow_html=True)


# --- 7. 사용자 입력 및 응답 생성 ---
if prompt := st.chat_input("시뮬레이션 변수를 입력하십시오."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("Client", avatar="👤"):
        st.markdown(f"<div class='lineblock'>{prompt}</div>", unsafe_allow_html=True)

    with st.spinner("Architect 시스템 연산 중..."):
        try:
            response_stream = st.session_state.chat.send_message(prompt, stream=True)
            with st.chat_message("Architect", avatar="🛡️"):
                response_placeholder = st.empty()
                full_response = ""
                for chunk in response_stream:
                    full_response += chunk.text
                    response_placeholder.markdown(f"<div class='lineblock'>{full_response}▌</div>", unsafe_allow_html=True)
                response_placeholder.markdown(f"<div class='lineblock'>{full_response}</div>", unsafe_allow_html=True)

            # ✅ 마지막 메시지일 때만 판례 분석 수행
            if any(word in prompt for word in ["결론", "최종", "최종 분석", "최종 결과", "마지막 단계"]):
                precedents, embeddings = load_and_embed_precedents("precedents_data.txt")
                similar_cases = find_similar_precedents(prompt, precedents, embeddings)
                if similar_cases:
                    st.markdown(f"<div class='lineblock'><b>🔎 실시간 판례 분석 결과</b></div>", unsafe_allow_html=True)
                    for case in similar_cases:
                        st.markdown(f"<div class='lineblock'>{case}</div>", unsafe_allow_html=True)

            st.session_state.messages.append({"role": "Architect", "content": full_response})
        except Exception as e:
            error_msg = f"시뮬레이션 오류 발생: {e}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "Architect", "content": error_msg})
