# 베리타스 엔진 8.1 — Auto-Analysis Mode + Dual RAG

import streamlit as st
import google.generativeai as genai
import os
import numpy as np
import re
import time
import json
import PyPDF2
from io import BytesIO

# ---------------------------------------
# 0. 기본 세팅
# ---------------------------------------
st.set_page_config(
    page_title="베리타스 엔진 8.1",
    page_icon="🛡️",
    layout="centered"
)

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

st.warning(
    "⚠️ 보안 경고: 본 시스템은 격리된 사설 환경(The Vault)에서 작동합니다. "
    "모든 데이터는 기밀로 취급되며 외부로 유출되지 않습니다."
)

# ---------------------------------------
# 1. API 키 설정
# ---------------------------------------
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
    if not API_KEY:
        raise ValueError("API Key is empty.")
    genai.configure(api_key=API_KEY)
except (KeyError, ValueError) as e:
    st.error(f"시스템 오류: 엔진 연결 실패. {e}")
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

@st.cache_data(show_spinner=True)
def load_and_embed_data(file_path: str, separator_regex: str = None):
    if not os.path.exists(file_path):
        print(f"[RAG] File not found: {file_path}")
        return [], []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"[RAG] Error reading file: {e}")
        return [], []

    if not content.strip():
        return [], []

    data_items = []
    embeddings = []

    if file_path.endswith(".jsonl"):
        for line in content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            txt = obj.get("rag_index") or obj.get("summary") or ""
            if not txt:
                continue

            emb = embed_text(txt, task_type="retrieval_document")
            if emb:
                data_items.append(obj)
                embeddings.append(emb)

    elif separator_regex:
        parts = re.split(separator_regex, content)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            emb = embed_text(p, task_type="retrieval_document")
            if emb:
                data_items.append({"rag_index": p, "raw_text": p})
                embeddings.append(emb)

    print(f"[RAG] Loaded {len(data_items)} items from {file_path}")
    return data_items, embeddings

def find_similar_items(query_text, items, embeddings, top_k=3, threshold=0.5):
    if not items or not embeddings:
        return []

    q_emb = embed_text(query_text, task_type="retrieval_query")
    if q_emb is None:
        return []

    sims = np.dot(np.array(embeddings), np.array(q_emb))
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
# 3. PDF 처리 함수
# ---------------------------------------
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- 페이지 {page_num + 1} ---\n"
                text += page_text
        
        return text
    
    except Exception as e:
        st.error(f"PDF 처리 실패: {e}")
        return None

def analyze_case_file(pdf_text: str, model):
    analysis_prompt = f"""
다음은 사건기록 PDF에서 추출한 내용입니다. 

[PDF 내용]
{pdf_text[:15000]}

[분석 지침]
1. 이 사건의 도메인 분류 (형사/민사/가사/행정/파산/IP/의료/세무 중 1개)
2. 세부 분야 (예: 형사-마약, 민사-계약분쟁 등)
3. 핵심 사실관계 5가지 (시간순 또는 중요도순)
4. 확보된 증거 목록 (문서명, 종류)
5. 피고인/원고 측 주장 요약
6. 상대방 측 주장 요약

반드시 아래 JSON 형식으로만 출력하세요. 다른 설명은 하지 마세요.

{{
  "domain": "형사",
  "subdomain": "마약",
  "key_facts": ["2023-05-01 필로폰 5g 소지로 체포", "경찰 조사 중 투약 인정", "초범", "생활비 목적 주장", "3개월간 10회 판매 정황"],
  "evidence": ["압수조서", "감정서(양성)", "카카오톡 대화 내역", "계좌이체 내역"],
  "our_claim": "단순 투약 목적이며 초범으로 선처 필요",
  "their_claim": "반복 판매로 영리 목적 인정"
}}
"""
    
    try:
        response = model.generate_content(analysis_prompt)
        result_text = response.text.strip()
        
        result_text = result_text.replace("```json", "").replace("```", "").strip()
        result = json.loads(result_text)
        return result
    
    except Exception as e:
        st.error(f"분석 실패: {e}")
        return None

# ---------------------------------------
# 4. 각종 유틸 함수
# ---------------------------------------
def _is_menu_input(s: str) -> bool:
    return bool(re.fullmatch(r"^\s*\d{1,2}(?:-\d{1,2})?\s*$", s))

def _is_final_report(txt: str) -> bool:
    return "전략 브리핑 보고서" in txt

def _query_title(prompt_text: str) -> str:
    return prompt_text[:67] + "..." if len(prompt_text) > 70 else prompt_text

def update_active_module(response_text: str):
    if "9." in response_text and "사건기록" in response_text and "Auto-Analysis" in response_text:
        st.session_state.active_module = "Auto-Analysis Mode"
        return
    
    m = re.search(r"'(.+?)' 모듈을 (?:최종 )?활성화합니다", response_text)
    if m:
        st.session_state.active_module = m.group(1).strip()
    elif "Phase 0" in response_text and not st.session_state.get("active_module"):
        st.session_state.active_module = "Phase 0"

# ---------------------------------------
# 5. 시스템 프라임 프롬프트 로드
# ---------------------------------------
try:
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        SYSTEM_INSTRUCTION = f.read()
    if len(SYSTEM_INSTRUCTION) < 100:
        raise ValueError("System prompt is too short.")
except (FileNotFoundError, ValueError) as e:
    st.error(f"치명적 오류: system_prompt.txt 로드 실패. {e}")
    st.stop()

# ---------------------------------------
# 6. 모델 & 세션 초기화
# ---------------------------------------
if "model" not in st.session_state:
    try:
        st.session_state.model = genai.GenerativeModel(
            "models/gemini-2.0-flash-exp",
            system_instruction=SYSTEM_INSTRUCTION,
        )
        st.session_state.chat = st.session_state.model.start_chat(history=[])
    except Exception as e:
        st.error(f"시스템 초기화 실패: {e}")
        st.stop()

    st.session_state.messages = []
    st.session_state.active_module = "Phase 0"

    st.session_state.precedents = []
    st.session_state.p_embeddings = []
    st.session_state.statutes = []
    st.session_state.s_embeddings = []

    try:
        init_prompt = "시스템 가동. Phase 0를 시작하라."
        resp = st.session_state.chat.send_message(init_prompt)
        init_text = resp.text
    except Exception as e:
        init_text = f"[시스템 초기화 실패: {e}]"

    st.session_state.messages.append({"role": "Architect", "content": init_text})
    update_active_module(init_text)

# ---------------------------------------
# 7. 과거 메시지 렌더링
# ---------------------------------------
for m in st.session_state.messages:
    role_name = "Client" if m["role"] == "user" else "Architect"
    avatar = "👤" if m["role"] == "user" else "🛡️"
    with st.chat_message(role_name, avatar=avatar):
        st.markdown(m["content"], unsafe_allow_html=True)

if st.session_state.messages:
    st.markdown(
        '<script>setTimeout(()=>{const el=window.parent.document.querySelector("section.main");if(el)el.scrollTop=el.scrollHeight},100)</script>',
        unsafe_allow_html=True
    )

# ---------------------------------------
# 8. PDF 업로드 UI (Auto-Analysis Mode 전용)
# ---------------------------------------
# ★★★ 조건: active_module이 "Auto-Analysis Mode"이고, 메시지가 2개 이상일 때만 표시 ★★★
if st.session_state.get("active_module") == "Auto-Analysis Mode" and len(st.session_state.messages) > 1:
    st.markdown("---")
    
    st.info("""
    **📄 사건기록 자동 분석 모드란?**
    
    PDF 파일(판결문, 고소장, 답변서 등)을 업로드하면 AI가 자동으로:
    - ✅ 사건 도메인 분류 (형사/민사/가사 등)
    - ✅ 핵심 사실관계 5가지 추출
    - ✅ 확보된 증거 목록 정리
    - ✅ 양측 주장 요약
    
    **처리 시간:** 약 1-3분 | **최대 크기:** 50MB | **형식:** 텍스트 기반 PDF만 가능
    """)
    
    st.subheader("📎 파일 업로드")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "사건기록 PDF를 선택하세요",
            type=["pdf"],
            help="판결문, 고소장, 답변서, 사건기록 등",
            label_visibility="collapsed"
        )
    
    with col2:
        if uploaded_file:
            st.metric("상태", "✅ 준비 완료", delta="업로드 완료")
        else:
            st.metric("상태", "⏳ 대기 중", delta="파일 선택")
    
    if uploaded_file is not None:
        file_size = uploaded_file.size / (1024 * 1024)
        
        with st.container():
            st.success(f"**파일명:** {uploaded_file.name}  |  **크기:** {file_size:.1f}MB")
        
        if st.button("🚀 자동 분석 시작", type="primary", use_container_width=True):
            with st.spinner("📄 PDF 텍스트 추출 중... (30초~2분 소요)"):
                pdf_text = extract_text_from_pdf(uploaded_file)
                
                if not pdf_text:
                    st.error("❌ PDF에서 텍스트를 추출할 수 없습니다.")
                    st.stop()
                
                st.success(f"✓ 텍스트 추출 완료 ({len(pdf_text):,} 글자)")
            
            with st.spinner("🧠 AI 분석 중... (1-2분 소요)"):
                analysis = analyze_case_file(pdf_text, st.session_state.model)
                
                if not analysis:
                    st.error("❌ 분석 실패. PDF 형식을 확인하고 다시 시도하세요.")
                    st.stop()
            
            st.success("✅ 분석 완료!")
            
            with st.expander("📊 분석 결과 상세 보기", expanded=True):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric("🏛️ 도메인", analysis["domain"])
                    st.metric("📌 세부 분야", analysis.get("subdomain", "미분류"))
                
                with col_b:
                    st.metric("📋 핵심 사실", f"{len(analysis.get('key_facts', []))}개")
                    st.metric("📂 증거 항목", f"{len(analysis.get('evidence', []))}개")
                
                st.markdown("---")
                st.markdown("**📌 핵심 사실관계**")
                for i, fact in enumerate(analysis.get("key_facts", []), 1):
                    st.markdown(f"{i}. {fact}")
                
                st.markdown("**📂 확보된 증거**")
                for i, ev in enumerate(analysis.get("evidence", []), 1):
                    st.markdown(f"{i}. {ev}")
                
                st.markdown("**⚖️ 양측 주장**")
                st.info(f"**우리 측:** {analysis.get('our_claim', '(정보 없음)')}")
                st.warning(f"**상대 측:** {analysis.get('their_claim', '(정보 없음)')}")
            
            domain_map = {
                "형사": "2",
                "민사": "8",
                "가사": "1",
                "이혼": "1",
                "파산": "3",
                "행정": "7",
                "세무": "6",
                "IP": "4",
                "의료": "5",
            }
            
            domain_num = domain_map.get(analysis["domain"], "8")
            
            st.info(
                f"💡 **다음 단계**\n\n"
                f"이 사건은 **{analysis['domain']}** 사건으로 분류되었습니다.\n\n"
                f"계속 진행하려면 아래 채팅창에 **{domain_num}**을 입력하세요."
            )
            
            st.session_state["auto_analysis"] = analysis
            st.session_state["pdf_text"] = pdf_text
    
    st.markdown("---")

# ---------------------------------------
# 9. 자동 분석 결과 활용 UI
# ---------------------------------------
if "auto_analysis" in st.session_state and st.session_state.get("active_module") != "Auto-Analysis Mode":
    auto_data = st.session_state["auto_analysis"]
    
    st.success(
        "💡 **자동 분석 결과가 감지되었습니다!**\n\n"
        "시스템이 변수 질문을 시작하면, 아래 버튼을 눌러 자동으로 답변할 수 있습니다."
    )
    
    if st.button("⚡ 자동 입력 활성화", type="secondary", use_container_width=True):
        auto_input = f"""
[자동 추출된 사건 정보]

**도메인:** {auto_data['domain']} - {auto_data.get('subdomain', '미분류')}

**핵심 사실관계:**
{chr(10).join(f"{i}. {fact}" for i, fact in enumerate(auto_data.get('key_facts', []), 1))}

**확보된 증거:**
{chr(10).join(f"- {ev}" for ev in auto_data.get('evidence', []))}

**우리 측 주장:**
{auto_data.get('our_claim', '(정보 없음)')}

**상대방 주장:**
{auto_data.get('their_claim', '(정보 없음)')}

위 정보를 바탕으로 시뮬레이션을 진행해주세요.
"""
        
        st.session_state.messages.append({"role": "user", "content": auto_input})
        del st.session_state["auto_analysis"]
        st.rerun()
    
    st.markdown("---")

# ---------------------------------------
# 10. 스트리밍 응답 함수
# ---------------------------------------
def stream_and_store_response(chat_session, prompt_to_send: str, spinner_text: str = "Architect 시스템 연산 중..."):
    full_response = ""
    start_time = time.time()

    with st.chat_message("Architect", avatar="🛡️"):
        placeholder = st.empty()
        try:
            with st.spinner(spinner_text):
                stream = chat_session.send_message(prompt_to_send, stream=True)
                for chunk in stream:
                    if not getattr(chunk, "parts", None):
                        full_response = "[시스템 경고: 응답이 안전 필터에 의해 차단되었습니다.]"
                        placeholder.error(full_response)
                        break
                    full_response += chunk.text
                    placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
            placeholder.markdown(full_response, unsafe_allow_html=True)
        except Exception as e:
            full_response = f"[치명적 오류: {e}]"
            placeholder.error(full_response)

    st.session_state.messages.append({"role": "Architect", "content": full_response})
    update_active_module(full_response)

    end_time = time.time()
    print(f"[LLM] 응답 시간: {end_time - start_time:.2f}s")
    return full_response

# ---------------------------------------
# 11. 메인 입력 루프
# ---------------------------------------
if prompt := st.chat_input("시뮬레이션 변수를 입력하십시오"):
    # ★★★ 9번 입력 감지 및 처리 ★★★
    if prompt.strip() == "9":
        st.session_state.active_module = "Auto-Analysis Mode"
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("Client", avatar="👤"):
            st.markdown(prompt, unsafe_allow_html=True)
        
        # AI 응답 받기
        response_text = stream_and_store_response(st.session_state.chat, prompt)
        
        # 화면 새로고침으로 PDF UI 표시
        st.rerun()
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("Client", avatar="👤"):
        st.markdown(prompt, unsafe_allow_html=True)

    is_data_ingestion_phase = "Phase 2" in (st.session_state.active_module or "")

    if (not st.session_state.statutes) and (not st.session_state.precedents):
        with st.spinner("분석 엔진(Dual RAG) 초기화 중..."):
            s_data, s_emb = load_and_embed_data("statutes_data.txt", r"\s*---END OF STATUTE---\s*")
            st.session_state.statutes = s_data
            st.session_state.s_embeddings = s_emb

            p_data, p_emb = load_and_embed_data("precedents_data.jsonl")
            if not p_data:
                p_data, p_emb = load_and_embed_data("precedents_data.txt", r"\s*---END OF PRECEDENT---\s*")
            st.session_state.precedents = p_data
            st.session_state.p_embeddings = p_emb

    rag_context = ""
    similar_precedents = []

    if not _is_menu_input(prompt) and not is_data_ingestion_phase:
        keywords = []
        
        domain_keywords = {
            "형사": "마약 필로폰 투약 매매 성범죄 강간 추행 음주운전 혈중알코올 도박",
            "민사": "계약 손해배상 채무 이행 해제 위약금",
            "가사": "이혼 양육권 재산분할 위자료 혼인",
            "파산": "파산 면책 채무 회생",
            "행정": "영업정지 과징금 처분 취소",
        }
        
        for domain, kw in domain_keywords.items():
            if domain in (st.session_state.active_module or ""):
                keywords.append(kw)
        
        keywords.append(prompt)
        contextual_query = " ".join(keywords)

        with st.spinner("실시간 데이터베이스 분석 중..."):
            if st.session_state.statutes:
                s_hits = find_similar_items(contextual_query, st.session_state.statutes, st.session_state.s_embeddings, top_k=3, threshold=0.55)
                if s_hits:
                    s_texts = [f"[유사도: {hit['similarity']:.2f}]\n{hit.get('rag_index', '내용 없음')}\n---\n" for hit in s_hits]
                    rag_context += "\n\n[시스템 참조: 검색된 관련 법령 데이터]\n" + "\n".join(s_texts)

            if st.session_state.precedents:
                similar_precedents = find_similar_items(contextual_query, st.session_state.precedents, st.session_state.p_embeddings, top_k=5, threshold=0.55)
                if similar_precedents:
                    p_texts = [f"[유사도: {hit['similarity']:.2f}]\n{hit.get('rag_index', '내용 없음')}\n---\n" for hit in similar_precedents]
                    rag_context += "\n\n[시스템 참조: 검색된 유사 판례 데이터]\n" + "\n".join(p_texts)

    final_prompt = f"[사용자 원문 입력]\n{prompt}\n{rag_context}"
    current_response = stream_and_store_response(st.session_state.chat, final_prompt)

    clean_response = re.sub("<[^<]+?>", "", current_response)

    if _is_final_report(clean_response) and similar_precedents:
        q_title = _query_title(prompt)
        st.markdown(f"**📚 실시간 판례 전문 분석 (P-RAG 결과)**\n\n* 검색 쿼리: `[{q_title}]`\n")

        for case_data in similar_precedents[:3]:
            sim_pct = int(round(case_data["similarity"] * 100))
            title = case_data.get("title", "제목 없음")
            case_no = case_data.get("case_no", case_data.get("id", ""))
            court = case_data.get("court", "")
            date = case_data.get("date", "")
            url = case_data.get("url")
            full_text = case_data.get("full_text", case_data.get("raw_text"))

            label = f"판례 [{title}]"
            if court and case_no:
                label += f" — {court} {case_no}"

            summary = case_data.get("rag_index", "요약 내용 없음")
            if len(summary) > 200:
                summary = summary[:197] + "..."

            link_md = f"[🔗 원문 링크 보기]({url})" if url else ""

            md = f"* **{label}**\n  - 선고: {date} | 유사도: {sim_pct}% | {link_md}\n  - 내용 요약: {summary}"
            st.markdown(md)

            if full_text:
                with st.expander("📄 판례 전문 보기"):
                    st.text(full_text)

    elif _is_final_report(clean_response) and not similar_precedents:
        st.info("ℹ️ 분석과 관련된 유사 판례가 데이터베이스에서 검색되지 않았습니다. (임계값 0.55)")
