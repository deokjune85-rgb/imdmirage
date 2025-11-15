import streamlit as st
import google.generativeai as genai
import requests
import numpy as np
from typing import List, Tuple

# -----------------------------
# 0. 기본 설정
# -----------------------------
API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=API_KEY)

RAW_URL = "https://raw.githubusercontent.com/deokjune85-rgb/imdmirage/main/precedents_data.txt"
EMBED_MODEL = "text-embedding-004"


# -----------------------------
# 1. 판례 로딩 + 임베딩
# -----------------------------
@st.cache_data(show_spinner="판례 데이터 불러오는 중...")
def load_precedents(raw_url: str) -> List[str]:
    resp = requests.get(raw_url, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"판례 원본을 불러오지 못했습니다 (status={resp.status_code}).")
    text = resp.text.strip()
    # 빈 줄(두 줄 이상 공백) 기준으로 판례 블록 분리
    blocks: List[str] = [b.strip() for b in text.split("\n\n") if b.strip()]
    return blocks


@st.cache_resource(show_spinner="판례 임베딩 계산 중...")
def embed_precedents(precedents: List[str]) -> np.ndarray:
    if not precedents:
        return np.zeros((0, 768), dtype=np.float32)

    embeddings: List[List[float]] = []
    for chunk in precedents:
        try:
            res = genai.embed_content(
                model=EMBED_MODEL,
                content=chunk,
            )
            emb = res["embedding"]
            embeddings.append(emb)
        except Exception:
            # 한 건 실패해도 전체가 죽지 않도록 스킵
            embeddings.append([0.0] * 768)

    return np.array(embeddings, dtype=np.float32)


def load_and_embed_precedents() -> Tuple[List[str], np.ndarray]:
    precedents = load_precedents(RAW_URL)
    embeddings = embed_precedents(precedents)
    return precedents, embeddings


# -----------------------------
# 2. 유사 판례 검색
# -----------------------------
def search_similar_cases(
    query: str,
    precedents: List[str],
    embeddings: np.ndarray,
    top_k: int = 5,
) -> List[Tuple[int, float, str]]:
    if embeddings.size == 0 or not precedents:
        return []

    # 질의 임베딩
    res = genai.embed_content(
        model=EMBED_MODEL,
        content=query,
    )
    q_emb = np.array(res["embedding"], dtype=np.float32)

    # cosine similarity
    norms = np.linalg.norm(embeddings, axis=1) * (np.linalg.norm(q_emb) + 1e-8)
    sims = embeddings @ q_emb / (norms + 1e-8)

    idx_scores = list(enumerate(sims.tolist()))
    idx_scores.sort(key=lambda x: x[1], reverse=True)
    idx_scores = idx_scores[:top_k]

    results: List[Tuple[int, float, str]] = []
    for idx, score in idx_scores:
        results.append((idx, score, precedents[idx]))
    return results


def build_rag_context(similar_cases: List[Tuple[int, float, str]]) -> str:
    if not similar_cases:
        return ""

    parts = []
    for rank, (idx, score, text) in enumerate(similar_cases, start=1):
        header = f"[유사 판례 {rank}] (score={score:.3f})\n"
        parts.append(header + text.strip())
    return "\n\n-----\n\n".join(parts)


# -----------------------------
# 3. LLM 호출
# -----------------------------
def call_llm(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp")
    resp = model.generate_content(prompt)
    return resp.text or ""


# -----------------------------
# 4. Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="IMD Mirage · 형사/민사 판례 RAG 엔진",
    layout="wide",
)

st.title("IMD Mirage · 형사/민사 판례 RAG 엔진")

st.markdown(
    """
사실관계와 고민을 아래에 적으면,  
내부 판례 데이터(RAG)를 검색해서 **유사 판례 + 종합 코멘트**를 생성합니다.
"""
)

# 세션 상태 초기화
if "precedents" not in st.session_state or "embeddings" not in st.session_state:
    with st.spinner("탄약고(RAG) 장전 중..."):
        precedents, embeddings = load_and_embed_precedents()
        st.session_state.precedents = precedents
        st.session_state.embeddings = embeddings

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
    st.write(f"판례 개수: {len(st.session_state.precedents)}건")
    st.write(f"임베딩 shape: {tuple(st.session_state.embeddings.shape)}")

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
2) RAG로 제공된 유사 판례를 요약·비교한 뒤,
3) 의뢰인이 실무에서 바로 쓸 수 있는 '실행 전략' 중심으로 정리하세요.
4) 감정 호소가 아니라, 객관적 자료·논리 구조 중심으로 설명합니다.
"""
        full_prompt = (
            system_guide
            + "\n\n[사건 개요]\n"
            + user_input.strip()
            + "\n\n[추가 요청]\n"
            + (extra_instr.strip() or "특이 요청 없음.")
            + "\n\n[내부 유사 판례 모음(RAG)]\n"
            + (rag_ctx or "유사 판례를 찾지 못했습니다. 일반적인 법리 중심으로 답변하세요.")
        )

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
