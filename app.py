import os
from typing import List, Tuple

import streamlit as st
import google.generativeai as genai
import requests
import numpy as np

# =============================
# 0. API 키 세팅 (근본 수정)
# =============================
API_KEY = os.environ.get("GOOGLE_API_KEY", "").strip()
try:
    if not API_KEY:
        API_KEY = str(st.secrets["GOOGLE_API_KEY"]).strip()
except Exception:
    pass

if not API_KEY:
    st.error("❌ GOOGLE_API_KEY 가 설정되어 있지 않습니다. 환경변수 또는 st.secrets 에 GOOGLE_API_KEY 를 넣어주세요.")
    st.stop()

genai.configure(api_key=API_KEY)

# 원격 txt (되면 쓰고, 안되면 그냥 RAG 없이 진행)
TXT_URL = "https://raw.githubusercontent.com/deokjune85-rgb/imdmirage/main/precedents_data.txt"

EMBED_MODEL = "models/text-embedding-004"
CHAT_MODEL = "models/gemini-1.5-pro"


# =============================
# 1. 판례 로딩 (에러 안 내고, 안 되면 빈 리스트)
# =============================
@st.cache_data(show_spinner="판례 데이터 불러오는 중...")
def load_precedents() -> List[str]:
    try:
        r = requests.get(TXT_URL, timeout=10)
        r.raise_for_status()
        raw = r.text.strip()
        if not raw:
            return []
        blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
        return blocks
    except Exception:
        # 네트워크 안 되면 그냥 RAG 없이 진행
        return []


# =============================
# 2. 판례 임베딩 (안 되면 0벡터)
# =============================
@st.cache_resource(show_spinner="판례 임베딩 계산 중...")
def embed_precedents(precedents: List[str]) -> np.ndarray:
    if not precedents:
        return np.zeros((0, 0), dtype=np.float32)

    embs: List[List[float]] = []
    dim = None

    for txt in precedents:
        try:
            res = genai.embed_content(
                model=EMBED_MODEL,
                content=txt,
            )
            vec = res["embedding"]
            if dim is None:
                dim = len(vec)
            embs.append(vec)
        except Exception:
            # 임베딩 실패해도 나머지는 진행
            if dim is None:
                dim = 768
            embs.append([0.0] * dim)

    return np.array(embs, dtype=np.float32)


def load_and_embed():
    precedents = load_precedents()
    embeddings = embed_precedents(precedents)
    return precedents, embeddings


# =============================
# 3. 유사 판례 검색 (실패해도 에러 없이 빈 리스트)
# =============================
def search_similar_cases(
    query: str,
    precedents: List[str],
    embeddings: np.ndarray,
    top_k: int = 5,
) -> List[Tuple[int, float, str]]:
    if embeddings.size == 0 or not precedents:
        return []

    try:
        q_res = genai.embed_content(
            model=EMBED_MODEL,
            content=query,
        )
        q_emb = np.array(q_res["embedding"], dtype=np.float32)

        if embeddings.shape[1] != q_emb.shape[0]:
            return []

        norms = np.linalg.norm(embeddings, axis=1) * (np.linalg.norm(q_emb) + 1e-8)
        sims = embeddings @ q_emb / (norms + 1e-8)

        idx_scores = list(enumerate(sims.tolist()))
        idx_scores.sort(key=lambda x: x[1], reverse=True)
        idx_scores = idx_scores[:top_k]

        results: List[Tuple[int, float, str]] = []
        for idx, score in idx_scores:
            results.append((idx, score, precedents[idx]))
        return results
    except Exception:
        return []


def build_rag_context(similar_cases: List[Tuple[int, float, str]]) -> str:
    if not similar_cases:
        return ""

    parts = []
    for rank, (idx, score, text) in enumerate(similar_cases, start=1):
        parts.append(f"[유사 판례 {rank}] (score={score:.3f})\n{text.strip()}")
    return "\n\n-----\n\n".join(parts)


# =============================
# 4. LLM 호출 (실패해도 문자열 리턴)
# =============================
def call_llm(prompt: str) -> str:
    try:
        model = genai.GenerativeModel(CHAT_MODEL)
        resp = model.generate_content(prompt)
        return (resp.text or "").strip()
    except Exception as e:
        return f"⚠️ LLM 호출 중 오류가 발생했습니다: {e}"


# =============================
# 5. Streamlit UI
# =============================
st.set_page_config(
    page_title="IMD Mirage · 형사/민사 판례 RAG 엔진",
    layout="wide",
)

st.title("IMD Mirage · 형사/민사 판례 RAG 엔진")

st.markdown(
    """
사실관계와 고민을 아래에 적으면,  
내부 판례 데이터(RAG)를 검색해서 **유사 판례 + 종합 코멘트**를 생성합니다.  
판례 데이터가 로딩되지 않으면, **일반 법리 + 전략 코멘트만** 생성합니다.
"""
)

# RAG 메모리 장전 (여기서 에러 안 나게 처리)
if "precedents" not in st.session_state or "embeddings" not in st.session_state:
    with st.spinner("탄약고(RAG) 장전 중..."):
        p, e = load_and_embed()
        st.session_state.precedents = p
        st.session_state.embeddings = e

col_left, col_right = st.columns([2, 1])

with col_left:
    user_input = st.text_area(
        "① 사실관계 / 사건 개요를 입력하세요.",
        height=220,
        placeholder="예) 2024. 5. 3. 밤 11시경, 술자리 이후 대리운전 호출했으나...",
    )

    extra_instr = st.text_area(
        "② 추가 요청(보고서 형식, 불기소 전략 강조 등)이 있으면 적어주세요.",
        height=120,
        placeholder="예) 불기소(혐의없음)를 1순위 목표로, 판례 인용을 중심으로 의견서 구조로 써줘.",
    )

    run_btn = st.button("⚖️ 판례 검색 + 전략 리포트 생성", type="primary")

with col_right:
    st.subheader("RAG 옵션")
    top_k = st.slider("유사 판례 개수", min_value=3, max_value=10, value=5, step=1)
    show_cases = st.checkbox("유사 판례 원문도 같이 보기", value=True)

    st.markdown("---")
    st.markdown("**RAG 상태**")

    precedents = st.session_state.precedents
    embeddings = st.session_state.embeddings

    st.write(f"판례 개수: {len(precedents)}건")
    st.write(f"임베딩 shape: {tuple(embeddings.shape)}")

    if not precedents:
        st.info("현재 원격 판례 데이터 로딩이 되지 않아, **RAG 없이 일반 법리 기반 답변만** 생성합니다.")

if run_btn and user_input.strip():
    precedents = st.session_state.precedents
    embeddings = st.session_state.embeddings

    with st.spinner("유사 판례 검색 및 리포트 생성 중..."):
        similar_cases = search_similar_cases(
            query=user_input,
            precedents=precedents,
            embeddings=embeddings,
            top_k=top_k,
        )
        rag_ctx = build_rag_context(similar_cases)

        system_guide = """
당신은 형사/민사 전문 변호사를 보조하는 AI 어시스턴트입니다.
1) 먼저 사건의 '핵심 쟁점'을 정리하고,
2) (있다면) RAG로 제공된 유사 판례를 요약·비교한 뒤,
3) 의뢰인이 실무에서 바로 쓸 수 있는 '실행 전략' 중심으로 정리하세요.
4) 감정 호소가 아니라, 객관적 자료·논리 구조 중심으로 설명합니다.
5) RAG 데이터가 없으면 일반적인 법리와 판례 경향을 바탕으로 답변합니다.
"""
        full_prompt = (
            system_guide
            + "\n\n[사건 개요]\n"
            + user_input.strip()
            + "\n\n[추가 요청]\n"
            + (extra_instr.strip() or "특이 요청 없음.")
        )

        if rag_ctx:
            full_prompt += "\n\n[내부 유사 판례 모음(RAG)]\n" + rag_ctx
        else:
            full_prompt += "\n\n[내부 유사 판례 모음(RAG)]\n활용 가능한 내부 판례 데이터가 없습니다. 일반적인 법리와 판례 경향에 근거해 답변하세요."

        answer = call_llm(full_prompt)

    st.subheader("🔎 종합 전략 리포트")
    st.write(answer)

    if show_cases and similar_cases:
        st.subheader("📚 참조된 유사 판례")
        for rank, (idx, score, text) in enumerate(similar_cases, start=1):
            with st.expander(f"유사 판례 {rank} (score={score:.3f})"):
                st.write(text)

elif run_btn and not user_input.strip():
    st.warning("사건 개요를 먼저 입력해주세요.")
