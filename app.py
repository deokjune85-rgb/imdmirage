# =====================================================
# 🛡️ 베리타스 엔진 7.6 — Contextual Dual RAG (JSONL/TXT Hybrid) + Relay Mechanism
# =====================================================
import streamlit as st
import google.generativeai as genai
import os
import numpy as np
import re
import time
import json  # ★ JSONL 처리를 위한 모듈 추가

# --- 1. 시스템 설정 및 CSS ---
st.set_page_config(page_title="베리타스 엔진 7.6", page_icon="🛡️", layout="centered")

# 'SaaS 삐끼' 새끼들의 '쓰레기' 'UI'를 '제거'하고 '폰트'를 '강제'한다.
custom_css = '''
<style>
#MainMenu, footer, header, .stDeployButton {visibility:hidden;}
html, body, div, span, p {
    font-family: "Noto Sans KR", sans-serif !important;
    font-size: 16px !important;
    line-height: 1.7 !important;
}
h1 { text-align: left !important; font-weight: 900 !important; font-size: 36px !important; margin-top: 10px !important; margin-bottom: 15px !important; }
strong, b { font-weight: 700; }
.fadein { animation: fadeInText 0.5s ease-in-out forwards; opacity: 0; }
@keyframes fadeInText { from {opacity: 0; transform: translateY(3px);} to {opacity: 1; transform: translateY(0);} }
[data-testid="stChatMessageContent"] { font-size: 16px !important; }
</style>
'''
st.markdown(custom_css, unsafe_allow_html=True)

# --- 2. 타이틀 및 경고 ---
st.title("베리타스 엔진 버전 7.6")
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

# --- [RAG 엔진 함수 정의] (★핵심 수정: JSONL/TXT 하이브리드 로더★) ---
EMBEDDING_MODEL_NAME = "models/text-embedding-004"


def embed_text(text, task_type="retrieval_document"):
    """'텍스트'를 '벡터(숫자)'로 '변환'하는 '연금술'."""
    try:
        clean_text = text.replace("\n", " ").strip()
        if not clean_text:
            return None
        result = genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=clean_text,
            task_type=task_type
        )
        return result["embedding"]
    except Exception as e:
        print(f"Embedding error: {e}")
        return None


@st.cache_data(show_spinner=True)  # '탄약고' '장전'은 '눈'으로 '확인'시켜준다.
def load_and_embed_data(file_path, separator_regex=None):
    """
    'JSONL'과 'TXT' '탄약고'를 '읽어' '벡터' '탄약'으로 '주조'한다.
    - 파일이 아예 없거나 읽기 실패: (None, None) 반환 → 진짜 '로드 실패'
    - 파일은 읽었는데 컨텐츠 없음: ([], []) 반환 → 파일은 정상, 데이터만 없음
    """
    # 1) 파일 존재 여부 체크
    if not os.path.exists(file_path):
        print(f"[RAG] File not found: {file_path}")
        return None, None  # ★ 여기서만 '진짜' 실패 취급

    # 2) 파일 읽기
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"[RAG] Error reading file {file_path}: {e}")
        return None, None  # ★ 읽기 자체가 안 되면 이것도 '진짜' 실패

    if not content.strip():
        print(f"[RAG] File {file_path} is empty.")
        return [], []  # 파일은 있으나 내용 없음

    data_items, embeddings = [], []

    # 3) JSONL 모드
    if file_path.endswith(".jsonl"):
        total_lines = 0
        parsed = 0
        embedded = 0

        for line_no, line in enumerate(content.strip().split("\n"), start=1):
            total_lines += 1
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
                parsed += 1
            except json.JSONDecodeError as e:
                print(f"[RAG][JSONL] Parse error {file_path}:{line_no} → {e}")
                continue

            # 'rag_index' 필드를 '임베딩' (핵심!)
            text_to_embed = item.get("rag_index")
            if not text_to_embed:
                print(f"[RAG][JSONL] Missing 'rag_index' at {file_path}:{line_no}")
                continue

            ebd = embed_text(text_to_embed, task_type="retrieval_document")
            if ebd:
                embeddings.append(ebd)
                data_items.append(item)  # 전체 객체 저장
                embedded += 1
            else:
                print(f"[RAG][JSONL] Embedding failed at {file_path}:{line_no}")

        print(
            f"[RAG][JSONL] {file_path} → lines={total_lines}, parsed={parsed}, embedded={embedded}"
        )

    # 4) TXT 모드 (법령 데이터 및 하위 호환성)
    elif separator_regex:
        chunks = re.split(separator_regex, content)
        raw_items = [p.strip() for p in chunks if p and p.strip()]
        print(f"[RAG][TXT] {file_path} → chunks={len(raw_items)}")
        for item_text in raw_items:
            ebd = embed_text(item_text, task_type="retrieval_document")
            if ebd:
                embeddings.append(ebd)
                data_items.append({"rag_index": item_text, "raw_text": item_text})

    print(f"[RAG] Loaded {len(data_items)} items from {file_path}.")
    return data_items, embeddings


def find_similar_items(query_text, items, embeddings, top_k=3, threshold=0.50):
    """'사건'과 '가장' '유사한' '총알' 3개를 '발사'한다."""
    if not embeddings or not items:
        return []
    q_emb = embed_text(query_text, task_type="retrieval_query")
    if q_emb is None:
        return []

    # 'NumPy'를 '사용'한 '벡터' '내적' '연산' (코사인 유사도)
    sims = np.dot(np.array(embeddings), np.array(q_emb))
    idxs = np.argsort(sims)[::-1][:top_k]

    results = []
    for i in idxs:
        if float(sims[i]) >= threshold:
            # '결과'에 '전체' '객체'와 '유사도'를 '저장'
            result_item = items[i].copy()
            result_item["similarity"] = float(sims[i])
            results.append(result_item)
    return results


# --- ★★★ '삭제'된 '유틸리티' '함수' '심장' '이식' ★★★ ---
def _is_menu_input(s: str) -> bool:
    """'입력'이 '단순' '숫자' '메뉴' '선택'인지 '판단'한다."""
    return bool(re.fullmatch(r"^\s*\d{1,2}(?:-\d{1,2})?\s*$", s))


def _is_final_report(txt: str) -> bool:
    """'응답'이 '최종 보고서' '형식'인지 '판단'한다."""
    return "전략 브리핑 보고서" in txt


def _query_title(prompt_text: str) -> str:
    """'RAG' '시각화'에 '사용'할 '쿼리' '제목'을 '추출'한다."""
    if len(prompt_text) > 70:
        return prompt_text[:67] + "..."
    return prompt_text


def update_active_module(response_text):
    """'뇌(EPE)'의 '응답'에서 '현재' '활성화'된 '모듈' '이름'을 '추출'한다."""
    match = re.search(r"\[(.+?)\]' 모듈을 활성화합니다", response_text)
    if match:
        st.session_state.active_module = match.group(1).strip()
    elif "Phase 0" in response_text:
        st.session_state.active_module = "Phase 0 (도메인 선택)"


# --- 4. 시스템 프라임 유전자 (Prime Genome) 로드 및 초기화 ---
try:
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        SYSTEM_INSTRUCTION = f.read()
    if len(SYSTEM_INSTRUCTION) < 100:
        raise ValueError("System prompt is too short.")
except (FileNotFoundError, ValueError) as e:
    st.error(f"치명적 오류: 시스템 코어(system_prompt.txt) 로드 실패. {e}")
    st.stop()

if "model" not in st.session_state:
    try:
        st.session_state.model = genai.GenerativeModel(
            "models/gemini-2.5-flash",
            system_instruction=SYSTEM_INSTRUCTION
        )

        # [★수정됨★] 듀얼 RAG 초기화 (JSONL + TXT)
        with st.spinner("분석 엔진(Dual RAG) 초기화 중... (최초 실행 시)"):
            # 1. 판례 데이터 로드 (P-RAG) - JSONL 우선, TXT 폴백
            p_data, p_emb = load_and_embed_data("precedents_data.jsonl")

            if p_data is None:
                # 진짜로 파일이 없거나 읽기 자체가 실패한 경우에만 폴백
                st.warning(
                    "경고: 'precedents_data.jsonl' 파일을 찾을 수 없거나 읽기 오류가 발생했습니다. 'txt' 파일로 폴백합니다."
                )
                p_data, p_emb = load_and_embed_data(
                    "precedents_data.txt",
                    r"\s*---END OF PRECEDENT---\s*"
                )
            elif isinstance(p_data, list) and len(p_data) == 0:
                # 파일은 읽었는데 유효한 판례가 0건인 경우 → 포맷/임베딩 문제일 가능성
                st.info(
                    "ℹ️ 'precedents_data.jsonl'은 로드되었으나 유효한 판례 항목이 없습니다. JSONL 형식과 'rag_index' 및 임베딩 오류 로그를 확인하세요."
                )

            st.session_state.precedents = p_data or []
            st.session_state.p_embeddings = p_emb or []

            # 2. 법령 데이터 로드 (S-RAG) - TXT 방식 유지 (없으면 그냥 빈 리스트)
            s_data, s_emb = load_and_embed_data(
                "statutes_data.txt",
                r"\s*---END OF STATUTE---\s*"
            )
            st.session_state.statutes = s_data or []
            st.session_state.s_embeddings = s_emb or []

        st.session_state.active_module = "초기 상태 (미정의)"

    except Exception as e:
        st.error(f"시스템 초기화 실패: {e}")
        st.stop()

# --- 5. 대화 세션 관리 및 자동 시작 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    # '초기' '프롬프트'를 '여기'서 '실행' (v7.5 교리)
    with st.spinner("Architect 시스템 가동..."):
        try:
            initial_prompt = (
                "시스템 가동. '동적 라우팅 프로토콜'을 실행하여 Phase 0를 시작하라."
            )
            chat = st.session_state.model.start_chat(history=[])
            response = chat.send_message(initial_prompt)
            st.session_state.messages.append(
                {"role": "Architect", "content": response.text}
            )
            st.session_state.chat = chat
            update_active_module(response.text)
        except Exception as e:
            st.error(f"시스템 초기화 실패: {e}")
            st.stop()

# --- 6. 대화 기록 표시 ---
for message in st.session_state.messages:
    role_name = "Client" if message["role"] == "user" else "Architect"
    avatar = "👤" if message["role"] == "user" else "🛡️"
    with st.chat_message(role_name, avatar=avatar):
        st.markdown(message["content"], unsafe_allow_html=True)

# --- 7. 입력 및 응답 생성 (★핵심 수정: JSONL 기반 출력 및 릴레이★) ---


def stream_and_store_response(
    chat_session,
    prompt_to_send,
    spinner_text="Architect 시스템 연산 중...",
):
    """'뇌(EPE)'에 '명령'을 '전송'하고, '응답'을 '실시간' '출력' 및 '저장'한다."""
    full_response = ""
    start_time = time.time()

    with st.chat_message("Architect", avatar="🛡️"):
        response_placeholder = st.empty()
        try:
            with st.spinner(spinner_text):
                response_stream = chat_session.send_message(
                    prompt_to_send,
                    stream=True
                )

                for chunk in response_stream:
                    # '안전' '필터'가 '작동'하면 '즉시' '중단'
                    if not chunk.parts:
                        full_response = (
                            "[시스템 경고: 응답이 '안전 필터'에 의해 '차단'되었습니다.]"
                        )
                        response_placeholder.error(full_response)
                        break

                    full_response += chunk.text
                    # '타이핑' '효과'
                    response_placeholder.markdown(
                        full_response + "▌",
                        unsafe_allow_html=True
                    )

            # '타이핑' '효과' '제거'
            response_placeholder.markdown(
                full_response,
                unsafe_allow_html=True
            )

        except Exception as e:
            full_response = f"[치명적 오류: {e}]"
            response_placeholder.error(full_response)

    # '세션'에 '최종' '응답' '저장'
    st.session_state.messages.append(
        {"role": "Architect", "content": full_response}
    )

    # 모듈 상태 갱신
    update_active_module(full_response)

    end_time = time.time()
    print(f"Response time: {end_time - start_time:.2f}s")
    return full_response


# 메인 입력 루프
if prompt := st.chat_input("시뮬레이션 변수를 입력하십시오."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("Client", avatar="👤"):
        st.markdown(prompt, unsafe_allow_html=True)

    # '날것(Raw Data)' '입력' '단계' '감지' (Phase 2)
    is_data_ingestion_phase = "Phase 2" in st.session_state.active_module

    # Contextual RAG 실행
    rag_context = ""
    similar_precedents = []

    # '메뉴' '선택'이 '아니'거나, 'Phase 2' '데이터' '입력'이 '아닐' '때'만 'RAG' '실행'
    if not _is_menu_input(prompt) and not is_data_ingestion_phase:

        # '컨텍스트' '쿼리' '생성'
        contextual_query = (
            f"현재 활성화된 모듈: {st.session_state.active_module}. "
            f"사용자 질문: {prompt}"
        )

        with st.spinner("실시간 데이터베이스 분석 중... (Dual RAG: 판례/법령)..."):
            # 1. 법령 검색 (S-RAG)
            if "statutes" in st.session_state and st.session_state.statutes:
                similar_statutes = find_similar_items(
                    contextual_query,
                    st.session_state.statutes,
                    st.session_state.s_embeddings,
                    top_k=3,
                    threshold=0.75,
                )
                if similar_statutes:
                    s_texts = [
                        f"[유사도: {c['similarity']:.2f}]\n"
                        f"{c.get('rag_index', '내용 없음')}\n---\n"
                        for c in similar_statutes
                    ]
                    rag_context += (
                        "\n\n[시스템 참조: 검색된 관련 법령 데이터]\n"
                        + "\n".join(s_texts)
                    )

            # 2. 판례 검색 (P-RAG)
            if "precedents" in st.session_state and st.session_state.precedents:
                similar_precedents = find_similar_items(
                    contextual_query,
                    st.session_state.precedents,
                    st.session_state.p_embeddings,
                    top_k=5,
                    threshold=0.75,
                )
                if similar_precedents:
                    p_texts = [
                        f"[유사도: {c['similarity']:.2f}]\n"
                        f"{c.get('rag_index', '내용 없음')}\n---\n"
                        for c in similar_precedents
                    ]
                    rag_context += (
                        "\n\n[시스템 참조: 검색된 유사 판례 데이터]\n"
                        + "\n".join(p_texts)
                    )

    # 최종 프롬프트 구성 및 시스템 응답 생성
    final_prompt = f"{prompt}\n{rag_context}"
    current_response = stream_and_store_response(
        st.session_state.chat,
        final_prompt
    )

    # [★핵심 수정★] 판례 시각화 및 원문 보기 기능 (JSONL 기반)
    clean_response = re.sub(
        "<[^<]+?>",
        "",
        current_response
    )  # 'HTML' '쓰레기' '제거'

    if _is_final_report(clean_response) and similar_precedents:
        q_title = _query_title(prompt)
        st.markdown(
            f"**📚 실시간 판례 전문 분석 (P-RAG 결과)**\n\n"
            f"* 검색 쿼리: `[{q_title}]`\n"
        )

        for case_data in similar_precedents[:3]:  # '상위' 3개만 '시각화'
            # 메타데이터 추출
            sim_pct = int(round(case_data["similarity"] * 100))
            title = case_data.get("title", "제목 없음")
            case_no = case_data.get("case_no", case_data.get("id", ""))
            court = case_data.get("court", "")
            date = case_data.get("date", "")
            url = case_data.get("url")
            full_text = case_data.get(
                "full_text",
                case_data.get("raw_text")
            )  # '전문' 또는 'TXT' '폴백'

            label = f"판례 [{title}]"
            if court and case_no:
                label += f" — {court} {case_no}"

            # 요약 카드
            summary = case_data.get("rag_index", "요약 내용 없음")
            if len(summary) > 200:
                summary = summary[:197] + "..."

            action_link = f"[🔗 원문 링크 보기]({url})" if url else ""

            item_md = (
                f"* **{label}**\n"
                f"  - 선고: {date} | 유사도: {sim_pct}% | {action_link}\n"
                f"  - 내용 요약 (RAG Index): {summary}"
            )
            st.markdown(item_md)

            # 원문 보기
            if full_text:
                with st.expander("📄 판례 전문 보기"):
                    st.text(full_text)

    elif (
        _is_final_report(clean_response)
        and not _is_menu_input(prompt)
        and not similar_precedents
    ):
        st.info(
            "ℹ️ 분석과 관련된 유사 판례가 데이터베이스에서 검색되지 않았습니다. (임계값 0.75)"
        )
