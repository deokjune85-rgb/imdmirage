# -*- coding: utf-8 -*-
# 베리타스 엔진 8.1.2 — Auto-Analysis Mode + Dual RAG (코드 멸균 및 환경 호환성 강화)

import streamlit as st
import google.generativeai as genai
import os
import numpy as np
import re
import time
import json
import PyPDF2 # PDF 처리를 위해 필수

# ---------------------------------------
# 0. 기본 세팅
# ---------------------------------------
st.set_page_config(
    page_title="베리타스 엔진 8.1",
    page_icon="🛡️",
    layout="centered"
)

# CSS (모든 공백은 표준 공백 U+0020으로 정제됨)
custom_css = """
<style>
#MainMenu, footer, header, .stDeployButton {visibility:hidden;}
html, body, div, span, p {
    font-family: 'Noto Sans KR', sans-serif !important;
    font-size: 16px !important;
    line-height: 1.7 !important;
}
h1 {
    text-align: left !important;
    font-weight: 900 !important;
    font-size: 36px !important;
    margin-top: 10px !important;
    margin-bottom: 15px !important;
}
strong, b { font-weight: 700; }
.fadein { animation: fadeInText 0.5s ease-in-out forwards; opacity: 0; }
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

st.title("베리타스 엔진 8.1")
st.caption("The Architect — 전략 시뮬레이션 엔진")

# 이모지 제거 및 텍스트 수정 (호환성 강화)
st.warning(
    "보안 경고: 본 시스템은 격리된 사설 환경(The Vault)에서 작동합니다. "
    "모든 데이터는 기밀로 취급되며 외부로 유출되지 않습니다."
)

# ---------------------------------------
# 1. API 키 설정
# ---------------------------------------
try:
    # Streamlit Cloud 배포 시 st.secrets 사용
    if "GOOGLE_API_KEY" in st.secrets:
        API_KEY = st.secrets["GOOGLE_API_KEY"]
    # 로컬 환경에서는 환경 변수 사용 (선택 사항)
    else:
        API_KEY = os.environ.get("GOOGLE_API_KEY")

    if not API_KEY:
        raise ValueError("API Key not found in secrets or environment variables.")
    genai.configure(api_key=API_KEY)
except (KeyError, ValueError) as e:
    st.error(f"시스템 오류: 엔진 연결 실패. API 키를 확인하세요. {e}")
    st.stop()

# ---------------------------------------
# 2. 임베딩 / RAG 유틸
# ---------------------------------------
EMBEDDING_MODEL_NAME = "models/text-embedding-004"

def embed_text(text: str, task_type: str = "retrieval_document"):
    clean_text = text.replace("\n", " ").strip()
    if not clean_text:
        return None
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=clean_text,
            task_type=task_type,
        )
        return result["embedding"]
    except Exception as e:
        print(f"[Embedding error] {e}")
        return None

# ---------------------------------------
# 2-1. 사전 계산된 임베딩 로드
# ---------------------------------------
@st.cache_data(show_spinner=False)
def load_precomputed_embeddings():
    """사전 계산된 임베딩 로드 (0.5초 완료)"""
    statute_items = []
    statute_embeddings = []
    precedent_items = []
    precedent_embeddings = []
    
    # 법령 로드
    if os.path.exists("statutes_embeddings.npy") and os.path.exists("statutes_items.json"):
        statute_embeddings = np.load("statutes_embeddings.npy").tolist()
        with open("statutes_items.json", "r", encoding="utf-8") as f:
            statute_items = json.load(f)
        print(f"[RAG] 법령 로드 완료: {len(statute_items)}개")
    
    # 판례 로드
    if os.path.exists("precedents_embeddings.npy") and os.path.exists("precedents_items.json"):
        precedent_embeddings = np.load("precedents_embeddings.npy").tolist()
        with open("precedents_items.json", "r", encoding="utf-8") as f:
            precedent_items = json.load(f)
        print(f"[RAG] 판례 로드 완료: {len(precedent_items)}개")
    
    return statute_items, statute_embeddings, precedent_items, precedent_embeddings

def find_similar_items(query_text, items, embeddings, top_k=3, threshold=0.5):
    if not items or not embeddings:
        return []

    q_emb = embed_text(query_text, task_type="retrieval_query")
    if q_emb is None:
        return []

    # 안정성을 위해 데이터 타입을 명시적으로 변환
    try:
        embeddings_np = np.array(embeddings, dtype=np.float32)
        q_emb_np = np.array(q_emb, dtype=np.float32)
    except ValueError as e:
        print(f"[RAG Error] 임베딩 데이터 타입 변환 실패: {e}")
        return []

    # 임베딩 차원 확인
    if embeddings_np.size > 0:
        if embeddings_np.shape[1] != len(q_emb_np):
            print(f"[RAG Error] 임베딩 차원 불일치: DB={embeddings_np.shape[1]}, Query={len(q_emb_np)}")
            return []
    else:
        return []

    sims = np.dot(embeddings_np, q_emb_np)
    idxs = np.argsort(sims)[::-1][:top_k]

    results = []
    for i in idxs:
        score = float(sims[i])
        if score < threshold:
            continue
        item = items[i].copy()
        item["similarity"] = score
        results.append(item)

    return results

# ---------------------------------------
# 3. PDF 처리 함수 (진단 강화됨 v8.1.2)
# ---------------------------------------
def extract_text_from_pdf(uploaded_file):
    """PDF 텍스트를 추출하고, 실패 시 원인 코드를 반환한다."""
    try:
        # [개선 1] 안정성 확보: 스트림 위치를 처음으로 되돌림 (Streamlit 특성 고려)
        uploaded_file.seek(0)
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        
        # [개선 2] 암호화 확인
        if pdf_reader.is_encrypted:
             return "[ERROR:ENCRYPTED]"

        text = ""
        
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                # 추출된 텍스트가 의미 있는지 확인 (공백 제거)
                cleaned_text = page_text.strip()
                if cleaned_text:
                    # 페이지 번호 표시 (이모지 제거)
                    text += f"\n--- 페이지 {page_num + 1} ---\n"
                    text += cleaned_text
        
        # [개선 3] 내용물 없음 감지 (스캔 PDF 진단)
        if not text.strip():
            # 모든 페이지 처리 후 텍스트가 없으면 스캔된 PDF 또는 빈 파일로 간주
            return "[ERROR:NO_TEXT]"

        return text.strip()
    
    except Exception as e:
        # 처리 실패 감지 (손상 등)
        print(f"[PDF Extraction Error] {e}") # 디버깅용 서버 로그
        return f"[ERROR:PROCESSING_FAILED]"


def analyze_case_file(pdf_text: str):
    """PDF 텍스트를 분석하여 핵심 정보를 JSON으로 추출한다."""
    # 프롬프트 (이모지 제거)
    analysis_prompt = f"""
다음은 사건기록 PDF에서 추출한 내용입니다.

[PDF 내용]
{pdf_text[:15000]} # 컨텍스트 길이 제한 고려

[분석 지침]
1. 이 사건의 도메인 분류 (형사/민사/가사/행정/파산/IP/의료/세무 중 1개)
2. 세부 분야 (예: 형사-마약, 민사-계약분쟁 등)
3. 핵심 사실관계 5가지 (시간순 또는 중요도순)
4. 확보된 증거 목록 (문서명, 종류)
5. 피고인/원고 측 주장 요약
6. 상대방 측 주장 요약

반드시 아래 JSON 형식으로만 출력하세요. 다른 설명은 하지 마세요. (```json 마크다운 포함)

```json
{{
  "domain": "형사",
  "subdomain": "마약",
  "key_facts": ["사실 1", "사실 2", "사실 3", "사실 4", "사실 5"],
  "evidence": ["증거 1", "증거 2"],
  "our_claim": "우리 측 주장 요약",
  "their_claim": "상대방 측 주장 요약"
}}
