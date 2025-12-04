# 베리타스 엔진 8.1 — Auto-Analysis Mode + Dual RAG (사전 임베딩)
# 베리타스 엔진 8.1 - 완전 수정판

import streamlit as st
import google.generativeai as genai
@@ -73,6 +73,7 @@
EMBEDDING_MODEL_NAME = "models/text-embedding-004"

def embed_text(text: str, task_type: str = "retrieval_document"):
    """텍스트 임베딩 생성"""
clean_text = text.replace("\n", " ").strip()
if not clean_text:
return None
@@ -87,6 +88,33 @@ def embed_text(text: str, task_type: str = "retrieval_document"):
print(f"[Embedding error] {e}")
return None

def find_similar_items(query_text, items, embeddings, top_k=3, threshold=0.5):
    """유사도 검색"""
    if not items or not embeddings:
        return []

    try:
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
    except Exception as e:
        print(f"[RAG Error] {e}")
        return []

# ---------------------------------------
# 2-1. 사전 계산된 임베딩 로드
# ---------------------------------------
@@ -98,49 +126,54 @@ def load_precomputed_embeddings():
precedent_items = []
precedent_embeddings = []

    # 법령 로드
    if os.path.exists("statutes_data.txt"):
        with open("statutes_data.txt", "r", encoding="utf-8") as f:
            content = f.read()
        
        parts = re.split(r"\s*---END OF STATUTE---\s*", content)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            emb = embed_text(p)
            if emb:
                statute_items.append({"rag_index": p, "raw_text": p})
                statute_embeddings.append(emb)
        
        print(f"[RAG] ✅ 법령 로드: {len(statute_items)}개")
    
    # 판례 로드
    if os.path.exists("precedents_data.jsonl"):
        with open("precedents_data.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    txt = obj.get("rag_index", "")
                    if txt:
                        emb = embed_text(txt)
                        if emb:
                            precedent_items.append(obj)
                            precedent_embeddings.append(emb)
                except:
    try:
        # 법령 로드
        if os.path.exists("statutes_data.txt"):
            with open("statutes_data.txt", "r", encoding="utf-8") as f:
                content = f.read()
            
            parts = re.split(r"\s*---END OF STATUTE---\s*", content)
            for p in parts:
                p = p.strip()
                if not p:
continue
                emb = embed_text(p)
                if emb:
                    statute_items.append({"rag_index": p, "raw_text": p})
                    statute_embeddings.append(emb)
            
            print(f"[RAG] 법령 로드: {len(statute_items)}개")

        print(f"[RAG] ✅ 판례 로드: {len(precedent_items)}개")
        # 판례 로드
        if os.path.exists("precedents_data.jsonl"):
            with open("precedents_data.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        txt = obj.get("rag_index", "")
                        if txt:
                            emb = embed_text(txt)
                            if emb:
                                precedent_items.append(obj)
                                precedent_embeddings.append(emb)
                    except:
                        continue
            
            print(f"[RAG] 판례 로드: {len(precedent_items)}개")
    
    except Exception as e:
        print(f"[RAG 로딩 에러] {e}")

return statute_items, statute_embeddings, precedent_items, precedent_embeddings

# ---------------------------------------
# 3. PDF 처리 함수
# ---------------------------------------
def extract_text_from_pdf(uploaded_file):
    """PDF에서 텍스트 추출"""
try:
pdf_reader = PyPDF2.PdfReader(uploaded_file)
text = ""
@@ -158,6 +191,7 @@ def extract_text_from_pdf(uploaded_file):
return None

def analyze_case_file(pdf_text: str, model):
    """PDF 내용 자동 분석"""
analysis_prompt = f"""
다음은 사건기록 PDF에서 추출한 내용입니다. 

@@ -200,27 +234,29 @@ def analyze_case_file(pdf_text: str, model):
# 4. 각종 유틸 함수
# ---------------------------------------
def _is_menu_input(s: str) -> bool:
    """메뉴 번호 입력 감지"""
return bool(re.fullmatch(r"^\s*\d{1,2}(?:-\d{1,2})?\s*$", s))

def _is_reset_keyword(s: str) -> bool:
    """처음으로/메인/초기화 키워드 감지"""
    """초기화 키워드 감지"""
keywords = ["처음", "메인", "초기화", "reset", "돌아가", "처음으로"]
return any(kw in s.lower() for kw in keywords)

def _is_final_report(txt: str) -> bool:
    """최종 보고서 생성 여부 확인"""
return "전략 브리핑 보고서" in txt

def _query_title(prompt_text: str) -> str:
    """쿼리 제목 생성"""
return prompt_text[:67] + "..." if len(prompt_text) > 70 else prompt_text

def update_active_module(response_text: str):
    # Auto-Analysis Mode 감지 (엄격한 조건)
    """활성 모듈 업데이트"""
if ("9." in response_text and "사건기록 자동 분석 모드" in response_text) or \
("Auto-Analysis Mode를 활성화합니다" in response_text):
st.session_state.active_module = "Auto-Analysis Mode"
return

    # 일반 모듈 활성화
m = re.search(r"'(.+?)' 모듈을 (?:최종 )?활성화합니다", response_text)
if m:
st.session_state.active_module = m.group(1).strip()
@@ -245,7 +281,7 @@ def update_active_module(response_text: str):
if "model" not in st.session_state:
try:
st.session_state.model = genai.GenerativeModel(
            "models/gemini-2.5-flash",
            "models/gemini-2.0-flash-exp",
system_instruction=SYSTEM_INSTRUCTION,
)
st.session_state.chat = st.session_state.model.start_chat(history=[])
@@ -280,127 +316,77 @@ def update_active_module(response_text: str):
with st.chat_message(role_name, avatar=avatar):
st.markdown(m["content"], unsafe_allow_html=True)

if st.session_state.messages:
    st.markdown(
        '<script>setTimeout(()=>{const el=window.parent.document.querySelector("section.main");if(el)el.scrollTop=el.scrollHeight},100)</script>',
        unsafe_allow_html=True
    )

# ---------------------------------------
# 8. PDF 업로드 UI (★★★ 핵심 수정 ★★★)
# 8. PDF 업로드 UI
# ---------------------------------------
# 조건: active_module이 정확히 "Auto-Analysis Mode"이고, 9번을 입력한 직후일 때만 표시
if st.session_state.get("active_module") == "Auto-Analysis Mode":
    # 마지막 사용자 메시지가 "9"인지 확인
last_user_msg = None
for m in reversed(st.session_state.messages):
if m["role"] == "user":
last_user_msg = m["content"].strip()
break

    # 9번 입력 직후에만 PDF UI 표시
if last_user_msg == "9":
st.markdown("---")

st.info("""
       **📄 사건기록 자동 분석 모드란?**
       
       PDF 파일(판결문, 고소장, 답변서 등)을 업로드하면 AI가 자동으로:
        - ✅ 사건 도메인 분류 (형사/민사/가사 등)
        - ✅ 핵심 사실관계 5가지 추출
        - ✅ 확보된 증거 목록 정리
        - ✅ 양측 주장 요약
        - 사건 도메인 분류 (형사/민사/가사 등)
        - 핵심 사실관계 5가지 추출
        - 확보된 증거 목록 정리
        - 양측 주장 요약
       
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
        uploaded_file = st.file_uploader(
            "사건기록 PDF를 선택하세요",
            type=["pdf"],
            help="판결문, 고소장, 답변서, 사건기록 등",
        )

if uploaded_file is not None:
file_size = uploaded_file.size / (1024 * 1024)
            
            with st.container():
                st.success(f"**파일명:** {uploaded_file.name}  |  **크기:** {file_size:.1f}MB")
            st.success(f"**파일명:** {uploaded_file.name}  |  **크기:** {file_size:.1f}MB")

if st.button("🚀 자동 분석 시작", type="primary", use_container_width=True):
                with st.spinner("📄 PDF 텍스트 추출 중... (30초~2분 소요)"):
                with st.spinner("📄 PDF 텍스트 추출 중..."):
pdf_text = extract_text_from_pdf(uploaded_file)

if not pdf_text:
                        st.error("❌ PDF에서 텍스트를 추출할 수 없습니다.")
                        st.error("PDF에서 텍스트를 추출할 수 없습니다.")
st.stop()

                    st.success(f"✓ 텍스트 추출 완료 ({len(pdf_text):,} 글자)")
                    st.success(f"텍스트 추출 완료 ({len(pdf_text):,} 글자)")

                with st.spinner("🧠 AI 분석 중... (1-2분 소요)"):
                with st.spinner("🧠 AI 분석 중..."):
analysis = analyze_case_file(pdf_text, st.session_state.model)

if not analysis:
                        st.error("❌ 분석 실패. PDF 형식을 확인하고 다시 시도하세요.")
                        st.error("분석 실패. PDF 형식을 확인하고 다시 시도하세요.")
st.stop()

                st.success("✅ 분석 완료!")
                st.success("분석 완료!")

with st.expander("📊 분석 결과 상세 보기", expanded=True):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.metric("🏛️ 도메인", analysis["domain"])
                        st.metric("📌 세부 분야", analysis.get("subdomain", "미분류"))
                    st.markdown(f"**도메인:** {analysis['domain']}")
                    st.markdown(f"**세부 분야:** {analysis.get('subdomain', '미분류')}")

                    with col_b:
                        st.metric("📋 핵심 사실", f"{len(analysis.get('key_facts', []))}개")
                        st.metric("📂 증거 항목", f"{len(analysis.get('evidence', []))}개")
                    
                    st.markdown("---")
                    st.markdown("**📌 핵심 사실관계**")
                    for i, fact in enumerate(analysis.get("key_facts", []), 1):
                    st.markdown("**핵심 사실관계:**")
                    for i, fact in enumerate(analysis.get('key_facts', []), 1):
st.markdown(f"{i}. {fact}")

                    st.markdown("**📂 확보된 증거**")
                    for i, ev in enumerate(analysis.get("evidence", []), 1):
                    st.markdown("**확보된 증거:**")
                    for i, ev in enumerate(analysis.get('evidence', []), 1):
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

@@ -412,12 +398,8 @@ def update_active_module(response_text: str):
if "auto_analysis" in st.session_state and st.session_state.get("active_module") != "Auto-Analysis Mode":
auto_data = st.session_state["auto_analysis"]

    st.success(
        "💡 **자동 분석 결과가 감지되었습니다!**\n\n"
        "AI가 자동으로 해당 모듈을 실행하여 완전한 보고서를 생성합니다."
    )
    st.success("AI가 자동으로 해당 모듈을 실행하여 완전한 보고서를 생성합니다.")

    # 자동 모듈 실행
domain_map = {
"형사": "2",
"민사": "8",
@@ -432,7 +414,6 @@ def update_active_module(response_text: str):

domain_num = domain_map.get(auto_data["domain"], "8")

    # 자동 입력 메시지 생성
auto_input = f"""
[자동 추출된 사건 정보]

@@ -450,25 +431,17 @@ def update_active_module(response_text: str):
상대방 주장:
{auto_data.get('their_claim', '(정보 없음)')}

위 정보를 바탕으로 {domain_num}번 모듈을 실행하여 완전한 전략 보고서를 생성하십시오.
위 정보를 바탕으로 완전한 전략 보고서를 생성하십시오.
"""

    # 메시지 추가
st.session_state.messages.append({"role": "user", "content": f"자동 분석 완료. {domain_num}번 모듈 실행"})

    with st.chat_message("Client", avatar="👤"):
        st.markdown(f"**자동 분석 결과를 바탕으로 {domain_num}번 모듈을 실행합니다.**")
    
    # AI에게 전송
    with st.spinner("완전한 전략 보고서 생성 중... (1-2분 소요)"):
    with st.spinner("완전한 전략 보고서 생성 중..."):
try:
            # 1단계: 도메인 번호 입력
resp1 = st.session_state.chat.send_message(domain_num)
st.session_state.messages.append({"role": "Architect", "content": resp1.text})

            # 형사인 경우 2-1 자동 입력
if domain_num == "2":
                # 세부 분야 매핑
subdomain_map = {
"마약": "2-1",
"성범죄": "2-2",
@@ -484,7 +457,6 @@ def update_active_module(response_text: str):
resp2 = st.session_state.chat.send_message(subdomain_num)
st.session_state.messages.append({"role": "Architect", "content": resp2.text})

            # 2단계: 자동 입력 데이터 전송
resp3 = st.session_state.chat.send_message(auto_input)

with st.chat_message("Architect", avatar="🛡️"):
@@ -495,30 +467,27 @@ def update_active_module(response_text: str):
except Exception as e:
st.error(f"자동 실행 실패: {e}")

    # 자동 분석 데이터 삭제
del st.session_state["auto_analysis"]
    
    st.markdown("---")
    st.rerun()

# ---------------------------------------
# 10. 스트리밍 응답 함수
# ---------------------------------------
def stream_and_store_response(chat_session, prompt_to_send: str, spinner_text: str = "Architect 시스템 연산 중..."):
def stream_and_store_response(chat_session, prompt_to_send: str):
    """스트리밍 응답 처리"""
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
            stream = chat_session.send_message(prompt_to_send, stream=True)
            for chunk in stream:
                if not getattr(chunk, "parts", None):
                    full_response = "[시스템 경고: 응답이 차단되었습니다.]"
                    placeholder.error(full_response)
                    break
                full_response += chunk.text
                placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
placeholder.markdown(full_response, unsafe_allow_html=True)
except Exception as e:
full_response = f"[치명적 오류: {e}]"
@@ -527,22 +496,19 @@ def stream_and_store_response(chat_session, prompt_to_send: str, spinner_text: s
st.session_state.messages.append({"role": "Architect", "content": full_response})
update_active_module(full_response)

    end_time = time.time()
    print(f"[LLM] 응답 시간: {end_time - start_time:.2f}s")
return full_response

# ---------------------------------------
# 11. 메인 입력 루프
# ---------------------------------------
if prompt := st.chat_input("시뮬레이션 변수를 입력하십시오"):

    # ★★★ 1. 초기화 키워드 감지 (최우선) ★★★
if _is_reset_keyword(prompt):
st.session_state.active_module = "Phase 0"
st.session_state.messages.append({"role": "user", "content": prompt})

with st.chat_message("Client", avatar="👤"):
            st.markdown(prompt, unsafe_allow_html=True)
            st.markdown(prompt)

reset_response = "시스템을 초기화합니다. Phase 0로 돌아갑니다."

@@ -551,7 +517,6 @@ def stream_and_store_response(chat_session, prompt_to_send: str, spinner_text: s

st.session_state.messages.append({"role": "Architect", "content": reset_response})

        # Phase 0 메뉴 다시 불러오기
try:
init_prompt = "시스템 가동. Phase 0를 시작하라."
resp = st.session_state.chat.send_message(init_prompt)
@@ -562,30 +527,35 @@ def stream_and_store_response(chat_session, prompt_to_send: str, spinner_text: s
st.session_state.messages.append({"role": "Architect", "content": init_text})
st.rerun()

    # ★★★ 2. 9번 입력 감지 ★★★
if prompt.strip() == "9":
st.session_state.active_module = "Auto-Analysis Mode"
st.session_state.messages.append({"role": "user", "content": prompt})

with st.chat_message("Client", avatar="👤"):
            st.markdown(prompt, unsafe_allow_html=True)
            st.markdown(prompt)

        response_text = stream_and_store_response(st.session_state.chat, prompt)
        stream_and_store_response(st.session_state.chat, prompt)
st.rerun()

st.session_state.messages.append({"role": "user", "content": prompt})
with st.chat_message("Client", avatar="👤"):
        st.markdown(prompt, unsafe_allow_html=True)
        st.markdown(prompt)

is_data_ingestion_phase = "Phase 2" in (st.session_state.active_module or "")

    # ★★★ RAG 초기화 (사전 임베딩 사용) - 경고 제거 ★★★
if (not st.session_state.statutes) and (not st.session_state.precedents):
        s_data, s_emb, p_data, p_emb = load_precomputed_embeddings()
        st.session_state.statutes = s_data
        st.session_state.s_embeddings = s_emb
        st.session_state.precedents = p_data
        st.session_state.p_embeddings = p_emb
        try:
            s_data, s_emb, p_data, p_emb = load_precomputed_embeddings()
            st.session_state.statutes = s_data
            st.session_state.s_embeddings = s_emb
            st.session_state.precedents = p_data
            st.session_state.p_embeddings = p_emb
        except Exception as e:
            print(f"[RAG 초기화 실패] {e}")
            st.session_state.statutes = []
            st.session_state.s_embeddings = []
            st.session_state.precedents = []
            st.session_state.p_embeddings = []

rag_context = ""
similar_precedents = []
@@ -640,13 +610,13 @@ def stream_and_store_response(chat_session, prompt_to_send: str, spinner_text: s

label = f"판례 [{title}]"
if court and case_no:
                label += f" — {court} {case_no}"
                label += f" - {court} {case_no}"

summary = case_data.get("rag_index", "요약 내용 없음")
if len(summary) > 200:
summary = summary[:197] + "..."

            link_md = f"[🔗 원문 링크 보기]({url})" if url else ""
            link_md = f"[원문 링크]({url})" if url else ""

md = f"* **{label}**\n  - 선고: {date} | 유사도: {sim_pct}% | {link_md}\n  - 내용 요약: {summary}"
st.markdown(md)
@@ -656,4 +626,4 @@ def stream_and_store_response(chat_session, prompt_to_send: str, spinner_text: s
st.text(full_text)

elif _is_final_report(clean_response) and not similar_precedents:
        st.info("ℹ️ 분석과 관련된 유사 판례가 데이터베이스에서 검색되지 않았습니다. (임계값 0.55)")
        st.info("분석과 관련된 유사 판례가 데이터베이스에서 검색되지 않았습니다. (임계값 0.55)")
