# 베리타스 엔진 8.0 — Domain 메뉴 + Dual RAG (TXT/JSONL 하이브리드)
# ======================================================
# 🛡️ 베리타스 엔진 8.1 — Domain 메뉴 개선 + Dual RAG (TXT/JSONL 하이브리드)
# ======================================================

import streamlit as st
import google.generativeai as genai
@@ -12,11 +14,12 @@
# 0. 기본 세팅
# ---------------------------------------
st.set_page_config(
    page_title="베리타스 엔진 8.0",
    page_title="베리타스 엔진 8.1",
page_icon="🛡️",
layout="centered"
)

# (CSS 내용 유지)
custom_css = """
<style>
#MainMenu, footer, header, .stDeployButton {visibility:hidden;}
@@ -46,7 +49,7 @@
st.markdown(custom_css, unsafe_allow_html=True)

# 상단 타이틀 + 경고
st.title("베리타스 엔진 8.0")
st.title("베리타스 엔진 8.1")
st.caption("Phase 0: 도메인 선택 → 이후 Architect가 자동 라우팅")

st.warning(
@@ -67,132 +70,81 @@
st.stop()

# ---------------------------------------
# 2. 임베딩 / RAG 유틸
# 2. 임베딩 / RAG 유틸 (기존 유지)
# ---------------------------------------
EMBEDDING_MODEL_NAME = "models/text-embedding-004"


def embed_text(text: str, task_type: str = "retrieval_document"):
clean_text = text.replace("\n", " ").strip()
    if not clean_text:
        return None
    if not clean_text: return None
try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=clean_text,
            task_type=task_type,
        )
        result = genai.embed_content(model=EMBEDDING_MODEL_NAME, content=clean_text, task_type=task_type)
return result["embedding"]
except Exception as e:
        print(f"[Embedding error] {e}")
        return None

        print(f"[Embedding error] {e}"); return None

@st.cache_data(show_spinner=True)
def load_and_embed_data(file_path: str, separator_regex: str = None):
    """
    - .jsonl: 줄 단위 JSON ➜ item['rag_index']를 임베딩
    - .txt  : separator_regex 기준으로 쪼개서 임베딩
    """
    if not os.path.exists(file_path):
        print(f"[RAG] File not found: {file_path}")
        return [], []

    if not os.path.exists(file_path): return [], []
try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"[RAG] Error reading file: {e}")
        return [], []

    if not content.strip():
        return [], []
        with open(file_path, "r", encoding="utf-8") as f: content = f.read()
    except Exception: return [], []
    if not content.strip(): return [], []

    data_items = []
    embeddings = []

    # JSONL
    data_items, embeddings = [], []
if file_path.endswith(".jsonl"):
for line in content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not line.strip(): continue
            try: obj = json.loads(line.strip())
            except json.JSONDecodeError: continue
txt = obj.get("rag_index") or obj.get("summary") or ""
            if not txt:
                continue

            if not txt: continue
emb = embed_text(txt, task_type="retrieval_document")
            if emb:
                data_items.append(obj)
                embeddings.append(emb)

    # TXT
            if emb: data_items.append(obj); embeddings.append(emb)
elif separator_regex:
parts = re.split(separator_regex, content)
for p in parts:
p = p.strip()
            if not p:
                continue
            if not p: continue
emb = embed_text(p, task_type="retrieval_document")
            if emb:
                data_items.append({"rag_index": p, "raw_text": p})
                embeddings.append(emb)

            if emb: data_items.append({"rag_index": p, "raw_text": p}); embeddings.append(emb)
print(f"[RAG] Loaded {len(data_items)} items from {file_path}")
return data_items, embeddings


def find_similar_items(query_text, items, embeddings, top_k=3, threshold=0.5):
    if not items or not embeddings:
        return []

    if not items or not embeddings: return []
q_emb = embed_text(query_text, task_type="retrieval_query")
    if q_emb is None:
        return []

    if q_emb is None: return []
sims = np.dot(np.array(embeddings), np.array(q_emb))
idxs = np.argsort(sims)[::-1][:top_k]

results = []
for i in idxs:
score = float(sims[i])
        if score < threshold:
            continue
        if score < threshold: continue
item = items[i].copy()
item["similarity"] = score
results.append(item)

return results


# ---------------------------------------
# 3. 각종 유틸 함수 (Phase 판단 등)
# 3. 각종 유틸 함수 (기존 유지)
# ---------------------------------------
def _is_menu_input(s: str) -> bool:
return bool(re.fullmatch(r"^\s*\d{1,2}(?:-\d{1,2})?\s*$", s))


def _is_final_report(txt: str) -> bool:
return "전략 브리핑 보고서" in txt


def _query_title(prompt_text: str) -> str:
return prompt_text[:67] + "..." if len(prompt_text) > 70 else prompt_text


def update_active_module(response_text: str):
    m = re.search(r"\[(.+?)\]' 모듈을 활성화합니다", response_text)
    m = re.search(r"'(.*?)' 모듈을 (?:최종 )?활성화합니다", response_text)
if m:
st.session_state.active_module = m.group(1).strip()
elif "Phase 0" in response_text and not st.session_state.get("active_module"):
st.session_state.active_module = "Phase 0 (도메인 선택)"


# ---------------------------------------
# 4. 시스템 프라임 프롬프트 로드
# ---------------------------------------
@@ -206,9 +158,12 @@ def update_active_module(response_text: str):
st.stop()

# ---------------------------------------
# 5. Phase 0 — 도메인 선택 UI (여기가 '선택지')
# 5. Phase 0 — 도메인 선택 UI (★수정됨★)
# ---------------------------------------
# [★수정됨★] "선택 안 함" 옵션 추가
DEFAULT_OPTION = "선택 안 함 (자동 판단)"
domain_options = [
    DEFAULT_OPTION,
"형사",
"민사",
"가사/이혼",
@@ -220,49 +175,58 @@ def update_active_module(response_text: str):
"기타(혼합)",
]

# 세션 상태 초기화 시 기본값 설정
if "selected_domain" not in st.session_state:
    st.session_state.selected_domain = "형사"
    st.session_state.selected_domain = DEFAULT_OPTION

st.subheader("Phase 0 — 사건 도메인 선택")

# 라디오 버튼 생성
selected_domain = st.radio(
    "현재 사건이 속한 주 도메인을 선택하세요.",
    "현재 사건이 속한 주 도메인을 선택하세요. (선택 안 함 시 시스템이 자동으로 판단합니다.)",
domain_options,
    # 현재 세션 상태에 따라 인덱스 설정
index=domain_options.index(st.session_state.selected_domain),
horizontal=True,
)

# 선택된 도메인 업데이트
st.session_state.selected_domain = selected_domain
st.info(f"현재 도메인: **{selected_domain}**")
st.info(f"현재 도메인 설정: **{selected_domain}**")

# ---------------------------------------
# 6. 모델 & 세션 초기화
# ---------------------------------------
if "model" not in st.session_state:
try:
st.session_state.model = genai.GenerativeModel(
            "models/gemini-2.5-flash",
            "models/gemini-2.5",
system_instruction=SYSTEM_INSTRUCTION,
)
st.session_state.chat = st.session_state.model.start_chat(history=[])
except Exception as e:
        st.error(f"시스템 초기화 실패: {e}")
        st.error(f"시스템 초기화 실패 (모델 로드 오류): {e}")
st.stop()

st.session_state.messages = []
st.session_state.active_module = f"Phase 0 — {selected_domain}"

    # RAG 코퍼스는 '지연 로딩' (처음 질문 들어올 때)
    # RAG 코퍼스 지연 로딩 설정
st.session_state.precedents = []
st.session_state.p_embeddings = []
st.session_state.statutes = []
st.session_state.s_embeddings = []

# 초기 인사/배치
try:
        # [★수정됨★] 초기 프롬프트에 도메인 정보 전달 방식 개선
        domain_info = selected_domain
        if selected_domain == DEFAULT_OPTION:
            domain_info = "미정의 (시스템 자동 판단 필요)"

init_prompt = (
            f"시스템 가동. 현재 선택된 도메인: {selected_domain}. "
            f"Phase 0에서 사건 구조를 스캔하고, 이후 Phase 1~를 동적으로 라우팅하라."
            f"시스템 가동. 현재 설정된 도메인: {domain_info}. "
            f"Phase 0에서 사건 구조를 스캔하고, 이후 Phase 1~를 동적으로 라우팅하라. 만약 도메인이 미정의라면, 사용자의 첫 입력을 분석하여 최적의 도메인을 판단하고 활성화하라."
)
resp = st.session_state.chat.send_message(init_prompt)
init_text = resp.text
@@ -273,7 +237,7 @@ def update_active_module(response_text: str):
update_active_module(init_text)

# ---------------------------------------
# 7. 과거 메시지 렌더링
# 7. 과거 메시지 렌더링 (자동 스크롤은 Streamlit 기본 기능)
# ---------------------------------------
for m in st.session_state.messages:
role_name = "Client" if m["role"] == "user" else "Architect"
@@ -282,38 +246,38 @@ def update_active_module(response_text: str):
st.markdown(m["content"], unsafe_allow_html=True)

# ---------------------------------------
# 8. 스트리밍 응답 함수
# 8. 스트리밍 응답 함수 (기존 유지 및 개선)
# ---------------------------------------
def stream_and_store_response(chat_session, prompt_to_send: str,
spinner_text: str = "Architect 시스템 연산 중..."):
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
                    # 응답 유효성 검사 강화
                    if not getattr(chunk, "parts", None) or not getattr(chunk, "text", None):
                        # 응답이 없거나 안전 필터에 막혔을 경우 처리
                        if not full_response: # 첫 응답이 막혔을 경우
                             full_response = "[시스템 경고: 응답 생성 실패 또는 안전 필터에 의해 차단됨.]"
placeholder.error(full_response)
break
full_response += chunk.text
placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
            placeholder.markdown(full_response, unsafe_allow_html=True)
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
# 9. 메인 입력 루프 + Dual RAG
# ---------------------------------------
@@ -323,7 +287,7 @@ def stream_and_store_response(chat_session, prompt_to_send: str,
with st.chat_message("Client", avatar="👤"):
st.markdown(prompt, unsafe_allow_html=True)

    # Phase 상태
    # Phase 상태 확인
is_data_ingestion_phase = "Phase 2" in (st.session_state.active_module or "")

# RAG 코퍼스 없으면 최초 1회 로딩
@@ -340,10 +304,7 @@ def stream_and_store_response(chat_session, prompt_to_send: str,
# 판례 JSONL → 없으면 TXT 폴백
p_data, p_emb = load_and_embed_data("precedents_data.jsonl")
if not p_data:
                st.warning(
                    "경고: 'precedents_data.jsonl' 로드 실패 또는 비어 있음. "
                    "'precedents_data.txt'로 폴백합니다."
                )
                # (경고 메시지 생략)
p_data, p_emb = load_and_embed_data(
"precedents_data.txt",
r"\s*---END OF PRECEDENT---\s*",
@@ -356,14 +317,19 @@ def stream_and_store_response(chat_session, prompt_to_send: str,
similar_precedents = []

if not _is_menu_input(prompt) and not is_data_ingestion_phase:
        # [★수정됨★] 도메인 정보 활용 개선
        current_domain = st.session_state.selected_domain
        if current_domain == DEFAULT_OPTION:
            current_domain = "미정의 (자동 판단 중)"

contextual_query = (
f"현재 활성화된 모듈: {st.session_state.active_module}. "
            f"선택된 도메인: {st.session_state.selected_domain}. "
            f"사용자 질문: {prompt}"
            f"선택된 도메인: {current_domain}. "
            f"사용자 질문/사실관계: {prompt}"
)

with st.spinner("실시간 데이터베이스 분석 중... (Dual RAG: 법령/판례)"):
            # 법령 검색
            # 법령 검색 (Threshold 0.75 유지)
if st.session_state.statutes:
s_hits = find_similar_items(
contextual_query,
@@ -383,7 +349,7 @@ def stream_and_store_response(chat_session, prompt_to_send: str,
"\n".join(s_texts)
)

            # 판례 검색
            # 판례 검색 (Threshold 0.75 유지)
if st.session_state.precedents:
similar_precedents = find_similar_items(
contextual_query,
@@ -403,18 +369,25 @@ def stream_and_store_response(chat_session, prompt_to_send: str,
"\n".join(p_texts)
)

    # 최종 프롬프트
    # 최종 프롬프트 구성
    # [★수정됨★] 도메인 정보 전달 방식 개선
    current_domain = st.session_state.selected_domain
    if current_domain == DEFAULT_OPTION:
        current_domain = "미정의 (시스템 자동 판단 필요)"

final_prompt = (
        f"[선택된 도메인] {st.session_state.selected_domain}\n"
        f"[현재 설정된 도메인] {current_domain}\n"
f"[사용자 원문 입력]\n{prompt}\n"
f"{rag_context}"
)
    
    # 시스템 응답 생성
current_response = stream_and_store_response(
st.session_state.chat,
final_prompt,
)

    # 판례 카드 시각화
    # 판례 카드 시각화 (기존 유지)
clean_response = re.sub("<[^<]+?>", "", current_response)

if _is_final_report(clean_response) and similar_precedents:
@@ -455,7 +428,7 @@ def stream_and_store_response(chat_session, prompt_to_send: str,
with st.expander("📄 판례 전문 보기"):
st.text(full_text)

    elif _is_final_report(clean_response) and not similar_precedents:
    elif _is_final_report(clean_response) and not similar_precedents and not _is_menu_input(prompt):
st.info(
"ℹ️ 분석과 관련된 유사 판례가 데이터베이스에서 검색되지 않았습니다. "
"(임계값 0.75)"
