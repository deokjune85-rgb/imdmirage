import streamlit as st
import google.generativeai as genai
import os
import numpy as np # RAG 엔진을 위한 벡터 연산 라이브러리

# --- 1. 시스템 설정 (The Vault & Mirage Protocol) ---
st.set_page_config(page_title="ARCHITECT 7.0", page_icon="🛡️", layout="centered")

# CSS 해킹 (신기루 프로토콜)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stDeployButton {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- 2. 타이틀 및 경고 (황제의 교리) ---
st.title("ARCHITECT 7.0 [Simulation Engine]")
st.error("보안 경고: 본 시스템은 격리된 사설 환경(The Vault)에서 작동합니다. 모든 데이터는 기밀로 취급되며 외부로 유출되지 않습니다.")

# --- 3. API 키 및 모델 설정 (The Engine & EPE/KB) ---
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
    if not API_KEY:
        raise ValueError("API Key is empty.")
    genai.configure(api_key=API_KEY)
except (KeyError, ValueError) as e:
    st.error(f"시스템 오류: 엔진 연결 실패. (API Key 누락 또는 비어있음): {e}")
    st.stop()

# --- [작전명: 트로이 목마] 게릴라 RAG 엔진 함수 정의 ---
EMBEDDING_MODEL_NAME = "models/text-embedding-004" # 구글 임베딩 모델

# 텍스트 임베딩 함수
def embed_text(text, task_type="retrieval_document"):
    try:
        # 텍스트 정제 (줄바꿈 제거 등)
        clean_text = text.replace('\n', ' ').strip()
        if not clean_text:
            return None
            
        result = genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=clean_text,
            task_type=task_type)
        return result['embedding']
    except Exception as e:
        print(f"Embedding error: {e}") # 콘솔 로그 기록
        return None

# 판례 데이터 로드 및 임베딩 함수 (st.cache_data로 캐싱하여 성능 최적화)
@st.cache_data
def load_and_embed_precedents(file_path='precedents_data.txt'):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}") # 콘솔 로그 기록
        return [], []
    
    # 파일 읽기 및 판례 분할
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}") # 콘솔 로그 기록
        return [], []
    
    precedents = content.split('---END OF PRECEDENT---')
    precedents = [p.strip() for p in precedents if p.strip()]
    
    # 각 판례 임베딩 (시간 소요)
    embeddings = []
    valid_precedents = []
    for precedent in precedents:
        embedding = embed_text(precedent)
        if embedding:
            embeddings.append(embedding)
            valid_precedents.append(precedent)
    
    print(f"Successfully loaded and embedded {len(valid_precedents)} precedents.") # 콘솔 로그 기록
    return valid_precedents, embeddings

# 유사 판례 검색 함수 (코사인 유사도)
def find_similar_precedents(query_text, precedents, embeddings, top_k=3):
    if not embeddings or not precedents:
        return []

    # 쿼리 임베딩
    query_embedding = embed_text(query_text, task_type="search_query")
    if query_embedding is None:
        return []
    
    # 코사인 유사도 계산 (NumPy 사용)
    # Google의 text-embedding-004는 정규화된 벡터를 반환하므로 내적(Dot product)이 코사인 유사도임.
    embeddings_np = np.array(embeddings)
    query_embedding_np = np.array(query_embedding)
    
    similarities = np.dot(embeddings_np, query_embedding_np)
    
    # 상위 K개 인덱스 찾기
    top_k_indices = np.argsort(similarities)[::-1][:top_k]
    
    # 결과 반환 (보고서 삽입용)
    results = []
    for idx in top_k_indices:
        # 유사도가 너무 낮으면 제외 (임계값 0.6 설정)
        if similarities[idx] > 0.6: 
            results.append(f"[유사 판례 발견 (유사도: {similarities[idx]:.2f})]\n{precedents[idx]}\n---\n")
    
    return results
# ------------------------------------------------------------


# 모델 설정: '프라임 게놈' 주입 (EPE/KB)
# 중요: 여기에 네놈의 '프라임 게놈' 전문을 넣어야 한다. (3단계의 RAG 지침이 포함된 버전으로!)
SYSTEM_INSTRUCTION = """
(여기에 프라임 게놈 전문 삽입 - 3단계의 RAG 지침 포함 확인!)
"""

# 모델 초기화 및 데이터 로드
if "model" not in st.session_state:
    try:
        # 1. 모델 로드
        st.session_state.model = genai.GenerativeModel('models/gemini-2.5-flash',
                                                       system_instruction=SYSTEM_INSTRUCTION)
        
        # 2. [트로이 목마] 판례 데이터 로드 및 임베딩 (캐시 사용)
        # 앱 시작 시 최초 1회 실행됨 (시간이 걸릴 수 있음)
        with st.spinner("판례 분석 엔진(RAG) 초기화 중... 데이터 임베딩 실행... (최초 실행 시)"):
            p, e = load_and_embed_precedents()
            st.session_state.precedents = p
            st.session_state.embeddings = e
            if not p:
                st.warning("경고: 판례 데이터(precedents_data.txt)를 로드하지 못했습니다. RAG 기능이 비활성화됩니다.")

    except Exception as e:
        st.error(f"시스템 초기화 실패 (모델 로드 또는 데이터 임베딩 오류): {e}")
        st.stop()

# --- 4. 대화 세션 관리 및 자동 시작 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat" not in st.session_state:
    if "model" in st.session_state:
        try:
            st.session_state.chat = st.session_state.model.start_chat(history=[])
            
            # 시스템 초기 메시지(Phase 0) 강제 생성
            initial_prompt = "시스템 가동. '동적 라우팅 프로토콜'을 실행하여 Phase 0를 시작하라."
            response = st.session_state.chat.send_message(initial_prompt)
            if response and response.text:
                st.session_state.messages.append({"role": "Architect", "content": response.text})

        except Exception as e:
            st.error(f"시스템 초기화 실패 (API 통신 오류): {e}")
    else:
        st.error("시스템 초기화 실패: 엔진 코어가 로드되지 않았습니다.")


# 이전 대화 기록 표시
for message in st.session_state.messages:
    role_name = message["role"]
    avatar = "🛡️"
    if role_name == "user":
        role_name = "Client"
        avatar = "👤"
        
    with st.chat_message(role_name, avatar=avatar):
        st.markdown(message["content"])

# --- 5. 사용자 입력 및 응답 생성 (RAG 통합) ---
if prompt := st.chat_input("시뮬레이션 변수를 입력하십시오."):
    if "chat" not in st.session_state:
        st.error("오류: 시뮬레이션 세션이 시작되지 않았습니다.")
        st.stop()

    # 사용자 입력 표시 및 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("Client", avatar="👤"):
        st.markdown(prompt)

    # [트로이 목마] RAG 실행 및 컨텍스트 생성
    rag_context = ""
    # 판례 데이터가 로드되어 있다면 RAG 실행
    if ("precedents" in st.session_state and st.session_state.precedents):
            with st.spinner("실시간 판례 데이터베이스 분석 중... 유사 사례 검색(RAG)..."):
                # 사용자의 입력을 쿼리로 사용하여 유사 판례 검색
                similar_precedents = find_similar_precedents(prompt, 
                                                                st.session_state.precedents, 
                                                                st.session_state.embeddings)
                if similar_precedents:
                    # 검색 결과를 시스템이 참조할 수 있도록 컨텍스트로 추가
                    rag_context = "\n\n[시스템 참조: 검색된 유사 판례 데이터]\n" + "\n".join(similar_precedents)

    # 최종 프롬프트 구성 (사용자 입력 + RAG 컨텍스트)
    # 시스템은 사용자 입력과 검색된 판례 데이터를 종합하여 응답을 생성함
    final_prompt = f"{prompt}\n{rag_context}"

    # 시스템 응답 생성 (API 호출)
    with st.spinner("Architect 시스템 연산 중... 변수 분석 및 시뮬레이션 실행..."):
        try:
            # 스트리밍 사용 (final_prompt 사용)
            response_stream = st.session_state.chat.send_message(final_prompt, stream=True)
            
            # 시스템 응답 표시 및 저장
            with st.chat_message("Architect", avatar="🛡️"):
                response_placeholder = st.empty()
                full_response = ""
                for chunk in response_stream:
                    if hasattr(chunk, 'text') and chunk.text:
                        full_response += chunk.text
                        response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "Architect", "content": full_response})
        
        except Exception as e:
            error_msg = f"시뮬레이션 오류 발생. 시스템 로그 확인 필요: {e}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "Architect", "content": error_msg})
