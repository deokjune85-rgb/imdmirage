# ======================================================
# 🛡️ 베리타스 엔진 7.7 — Contextual Dual RAG + File Upload + Relay Mechanism
# ======================================================
import streamlit as st
import google.generativeai as genai
import os
import numpy as np
import re
import time
import json
# ★신규 임포트★
from pdfminer.high_level import extract_text
import io

# --- 1. 시스템 설정 및 CSS (기존 유지) ---
st.set_page_config(page_title="베리타스 엔진 7.7", page_icon="🛡️", layout="centered")

# (CSS 내용 생략 - 이전 버전과 동일하게 유지하되, color: #FFFFFF 강제는 제거되었는지 확인)
custom_css = """<style>...</style>""" 
st.markdown(custom_css, unsafe_allow_html=True)

# --- 2. 타이틀 및 경고 ---
st.title("베리타스 엔진 버전 7.7")
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

# --- [RAG 엔진 함수 정의] (기존 유지) ---
# (embed_text, load_and_embed_data, find_similar_items, _parse_precedent_block 함수 유지 - 생략)
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
def embed_text(text, task_type="retrieval_document"): ...
@st.cache_data
def load_and_embed_data(file_path, separator_regex=None): ...
def find_similar_items(query_text, items, embeddings, top_k=3, threshold=0.50): ...
def _parse_precedent_block(text: str) -> dict: ...

# (유틸리티 함수 유지)
def _is_menu_input(s: str) -> bool: ...
def _is_final_report(txt: str) -> bool: ...
def _query_title(prompt_text: str) -> str: ...
def update_active_module(response_text): ...

# [★신설★] 파일 처리 함수
def process_uploaded_file(uploaded_file):
    """업로드된 파일(TXT, PDF)에서 텍스트를 추출한다."""
    text = ""
    try:
        if uploaded_file.type == "text/plain":
            # TXT 파일 읽기
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8", errors="ignore"))
            text = stringio.read()
        elif uploaded_file.type == "application/pdf":
            # PDF 파일 처리 (pdfminer 사용)
            # 주의: 이것은 텍스트 기반 PDF만 처리 가능하며, 이미지 기반 PDF는 실패함.
            bytes_data = uploaded_file.read()
            text = extract_text(io.BytesIO(bytes_data))
            
            if not text.strip():
                # 텍스트 추출 실패 시 (이미지 기반 PDF 등) - MSP 셀링 포인트
                return "[파일 처리 오류: PDF에 추출 가능한 텍스트가 없습니다. 이미지 기반 PDF(스캔본 등)는 현재 데모에서 지원되지 않습니다. (MSP 버전에서 고성능 OCR 엔진으로 지원 예정)]"
        else:
            return "[파일 처리 오류: 지원되지 않는 파일 형식입니다. (TXT, PDF만 가능)]"
        
        # 최대 길이 제한 (안정성 확보)
        MAX_LEN = 50000
        if len(text) > MAX_LEN:
            text = text[:MAX_LEN] + f"\n\n[...내용 생략됨 (최대 길이 {MAX_LEN}자 초과)...]"
        
        return text

    except Exception as e:
        return f"[파일 처리 오류: {e}]"


# --- 4. 시스템 프라임 유전자 로드 및 초기화 ---
# ... (system_prompt.txt 로드 및 모델/RAG 초기화 로직 유지 - 생략) ...

# --- 5. 대화 세션 관리 및 자동 시작 ---
# ... (생략) ...

# --- 6. 대화 출력 ---
# ... (생략) ...

# 스트리밍 출력 및 저장 함수 (기존 유지)
def stream_and_store_response(chat_session, prompt_to_send, spinner_text="Architect 시스템 연산 중..."):
    # ... (함수 내용 유지 - 생략) ...


# --- 7. 입력 및 응답 생성 (★핵심 수정: 파일 업로드 통합 + 릴레이★) ---

# [★핵심 수정 1: Phase 2 상태 감지★]
# 입력 처리 전에 Phase 2 상태를 먼저 확인하여 UI를 결정한다.
is_phase2_active = False
if st.session_state.get("messages"): # 세션 상태 접근 방식 개선
    last_architect_msg = ""
    # 마지막 Architect 메시지 찾기
    for msg in reversed(st.session_state.messages):
        if msg['role'] == 'Architect':
            last_architect_msg = re.sub('<[^<]+?>', '', msg['content']); break
    
    # 이전 메시지가 Phase 2 데이터 요청이었는지 확인 (키워드 기반 감지)
    if "Phase 2:" in last_architect_msg and ("데이터를 지금 시스템에 입력하십시오" in last_architect_msg or "엔진'을 가동하여" in last_architect_msg):
        is_phase2_active = True

# [★핵심 수정 2: 조건부 파일 업로드 UI 표시★]
uploaded_file = None
input_text = None # 처리할 최종 텍스트

if is_phase2_active:
    st.info("📂 Phase 2 활성화: 분석할 증거 데이터(TXT 또는 텍스트 기반 PDF)를 업로드하거나, 아래 채팅창에 텍스트를 직접 입력하십시오.")
    # key를 사용하여 파일 업로더 상태를 명확히 관리
    uploaded_file = st.file_uploader("증거 파일 업로드", type=['txt', 'pdf'], key="phase2_uploader")

# 메인 입력 루프
chat_prompt = st.chat_input("시뮬레이션 변수를 입력하십시오.")

# [★핵심 수정 3: 입력 소스 결정★]

# 1. 파일이 업로드되었으면 파일 내용을 우선 사용 (Streamlit은 파일 업로드 시 자동으로 스크립트를 재실행함)
if uploaded_file is not None:
    with st.spinner("업로드된 파일 처리 중... 텍스트 추출..."):
        input_text = process_uploaded_file(uploaded_file)
    # 파일 처리 후에는 input_text가 채워진 상태로 아래 로직을 진행함.

# 2. 채팅창에 입력이 있고, 파일 업로드가 처리되지 않은 경우
elif chat_prompt:
    input_text = chat_prompt

# 처리할 입력이 있을 경우 실행 (Prompt 변수명 통일)
prompt = input_text

if prompt:
    # 사용자 입력 표시 (요약해서 표시)
    display_text = prompt
    if len(display_text) > 500:
        display_text = display_text[:500] + f"...(내용 생략됨: 총 {len(prompt)}자)..."
        
    st.session_state.messages.append({"role": "user", "content": f"<div class='fadein'>{display_text}</div>"})
    with st.chat_message("Client", avatar="👤"):
        st.markdown(f"<div class='fadein'>{display_text}</div>", unsafe_allow_html=True)

    # (이하 Contextual RAG 실행, 릴레이 메커니즘, 판례 시각화 로직은 이전 버전(7.6)과 동일하게 유지)
    # ... (여기에 이전 버전의 RAG 실행, 응답 생성, 릴레이 로직을 그대로 붙여넣어라) ...
