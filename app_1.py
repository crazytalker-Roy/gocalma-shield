# Before running this app, execute in a terminal:
# 1) pip install -r requirements.txt
# 2) python -m spacy download en_core_web_lg
# 3) python -m spacy download zh_core_web_lg
# 4) streamlit run app_1.py

import re
import html
import json
import io
import hashlib
from typing import List

import streamlit as st
import streamlit.components.v1 as components
import requests
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

# ===========================
# Global / cached: NLP & Analyzer initialization
# ===========================

@st.cache_resource(show_spinner=False)
def get_presidio_analyzer() -> AnalyzerEngine:
    """
    Build a Presidio AnalyzerEngine supporting both English and Chinese.
    Uses spaCy en_core_web_lg / zh_core_web_lg (Large models) for better accuracy offline.
    """
    config = {
        "nlp_engine_name": "spacy",
        "models":[
            {"lang_code": "en", "model_name": "en_core_web_lg"},
            {"lang_code": "zh", "model_name": "zh_core_web_lg"},
        ],
    }
    try:
        provider = NlpEngineProvider(nlp_configuration=config)
    except TypeError:
        provider = NlpEngineProvider(conf=config)

    try:
        nlp_engine = provider.create_engine()
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize NLP engine. Please ensure spaCy models are downloaded:\n"
            "  python -m spacy download en_core_web_lg\n"
            "  python -m spacy download zh_core_web_lg\n"
            f"Original error: {e}"
        ) from e

    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=["en", "zh"],
    )

    _ = AnonymizerEngine()
    return analyzer

# ===========================
# File upload helpers (txt/pdf)
# ===========================

def decode_bytes_safely(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk", "latin-1"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode("utf-8", errors="replace")

def extract_text_from_uploaded_file(uploaded_file) -> str:
    filename = (uploaded_file.name or "").lower()
    data = uploaded_file.getvalue()

    if filename.endswith(".pdf"):
        if PdfReader is None:
            raise RuntimeError("PDF support dependency is not installed. Please run: pip install pypdf")
        reader = PdfReader(io.BytesIO(data))
        pages =[]
        for p in reader.pages:
            t = p.extract_text() or ""
            if t.strip():
                pages.append(t)
        return "\n\n".join(pages).strip()

    return decode_bytes_safely(data).strip()

# ===========================
# Language detection & mapping
# ===========================

def detect_lang_auto(text: str) -> str:
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    return "en"

LANG_OPTIONS =["English", "简体中文", "Auto/Mixed"]
LANG_TO_CODE = {
    "English": "en",
    "简体中文": "zh",
    "Auto/Mixed": "auto",
}

# ===========================
# Custom masking / anonymization helpers
# ===========================

# ===========================
# Format-Preserving Masking Functions
# ===========================
def mask_email(email: str) -> str:
    """Email: john.smith99@gmail.com -> j****9@gmail.com"""
    if "@" not in email:
        return email
    local, domain = email.split("@", 1)
    if len(local) <= 1:
        new_local = "*"
    elif len(local) == 2:
        new_local = local[0] + "*"
    else:
        new_local = f"{local[0]}****{local[-1]}"
    return f"{new_local}@{domain}"

def mask_phone(phone: str) -> str:
    """Phone: +1-555-123-4567 -> +1-55*-***-4567 (keeps format, masks middle)"""
    digits = [c for c in phone if c.isdigit()]
    if len(digits) < 7:
        return re.sub(r"\d", "*", phone)
    
    res =[]
    d_idx = 0
    for c in phone:
        if c.isdigit():
            # Keep first 3 digits and last 4 digits
            if d_idx < 3 or d_idx >= len(digits) - 4:
                res.append(c)
            else:
                res.append('*')
            d_idx += 1
        else:
            res.append(c)
    return "".join(res)

def mask_id_like(id_str: str) -> str:
    """IDs/Cards: preserves dashes/spaces, replaces middle alphanumerics with '*'"""
    total_alnum = sum(1 for c in id_str if c.isalnum())
    
    # 🛡️ Fragment Protection Mechanism
    # If the input string has <=4 alphanumeric characters remaining (e.g., tail digits 4321),
    # return it as is to prevent NLP models from re-masking it into ****.
    if total_alnum <= 4:
        return id_str
        
    keep_front = 2 if total_alnum <= 9 else 4
    keep_back = 2 if total_alnum <= 9 else 4
    
    res =[]
    idx = 0
    for c in id_str:
        if c.isalnum():
            if idx < keep_front or idx >= total_alnum - keep_back:
                res.append(c)
            else:
                res.append('*')
            idx += 1
        else:
            res.append(c)
    return "".join(res)

def mask_person(name: str) -> str:
    name = name.strip()
    if not name: return name
    if bool(re.search(r"[\u4e00-\u9fff]", name)):
        length = len(name)
        if length <= 2: return name[0] + "*"
        elif length == 3: return name[:2] + "*"
        else: return name[:2] + "*" * (length - 2)
    else:
        parts = name.split()
        return " ".join([p[0] + "***" for p in parts if p])

def _mask_address_en_zh(text: str) -> str:
    # English: Precisely match street names starting with digits (e.g., 123 Main St -> **** Main St)
    text = re.sub(r"\b\d{1,5}\b(?=\s+[A-Z][a-z]+)", "****", text)
    # English: Erase room/unit numbers (e.g., Suite 105 -> Suite ****)
    text = re.sub(r"(?i)\b(Suite|Apt|Unit|Room)\s+[\w-]+\b", r"\1 ****", text)
    
    # Chinese: Mask standard building, unit, and room numbers
    text = re.sub(r"\d{1,3}号楼\d{1,3}单元\d{1,4}", "*号楼*单元****", text)
    text = re.sub(r"\d{1,3}号楼\d{1,4}(?:室|房)?", "*号楼****", text)
    text = re.sub(r"\d{1,3}号楼", "*号楼", text)
    text = re.sub(r"\d{1,3}单元", "*单元", text)
    text = re.sub(r"\d{2,4}(室|房)", r"****\1", text)
    text = re.sub(r"\d{1,4}号", "*号", text)
    text = re.sub(r"([\u4e00-\u9fff]{2,10})\d{3,4}(?=$|[，,。\s])", r"\1****", text)
    
    return text

def apply_custom_mask(entity_type: str, value: str) -> str:
    et = entity_type.upper()
    
    # 🎯 Whitelist Mechanism: Only apply masking to specific PII types
    if et == "PHONE_NUMBER": return mask_phone(value)
    if et == "PERSON": return mask_person(value)
    if et == "EMAIL_ADDRESS": return mask_email(value)
    
    id_types = {
        "ID_CARD", "ID_NUMBER", "PASSPORT", "US_PASSPORT", "US_SSN", 
        "US_ITIN", "US_DRIVER_LICENSE", "UK_NHS", "CREDIT_CARD", "IBAN_CODE"
    }
    if et in id_types:
        return mask_id_like(value)

    return value

# ===========================
# Regex Pre-Pass 
# ===========================
def _regex_pre_pass(text: str) -> str:
    """在 Presidio 启动前，使用精准正则拦截格式化数据，防止 NLP 模型产生幻觉和遗漏"""
    # 1. Password and username masking
    text = re.sub(
        r"((?:原密码|新密码|密码|old password|new password|password)\s*[:：])\s*[^\s，,。]+",
        r"\1[REDACTED_PASSWORD]",
        text,
        flags=re.IGNORECASE,
    )

    # 2. Person name masking
    text = re.sub(
        r"((?:法人|用户|姓名|联系人|客户|真实姓名)\s*[:：]\s*)([\u4e00-\u9fa5]{2,4})(?=$|[，,\s。])", 
        lambda m: m.group(1) + mask_person(m.group(2)), 
        text
    )

    # 3. IP / MAC
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[REDACTED_IP]", text)
    text = re.sub(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b", "[REDACTED_MAC]", text)
    
    # 4. Email
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", lambda m: mask_email(m.group(0)), text)
    
    # 5. SSN (000-00-0000 -> ***-**-0000)
    text = re.sub(r"\b\d{3}-\d{2}-(\d{4})\b", r"***-**-\1", text)
    
    # 6. Credit Cards
    text = re.sub(r"\b(?:\d{4}[-\s]?){3}\d{4}\b", lambda m: mask_id_like(m.group(0)), text)
    
    # 7. Phones (US & Intl format)
    text = re.sub(r"\+?\d{1,3}[-.\s]\d{3}[-.\s]\d{3}[-.\s]\d{4}\b", lambda m: mask_phone(m.group(0)), text)
    text = re.sub(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b", lambda m: mask_phone(m.group(0)), text)
    
    # 8. Chinese IDs & Phones
    text = re.sub(r"\b\d{17}[\dXx]\b", lambda m: mask_id_like(m.group(0)), text)
    text = re.sub(r"\b[0-9A-Z]{18}\b", lambda m: mask_id_like(m.group(0)), text)
    text = re.sub(r"\b1\d{10}\b", lambda m: mask_phone(m.group(0)), text)
    
    # 9. Passports & Driver Licenses (Fallback catch for 7-11 Alphanumeric upper cases)
    text = re.sub(r"\b[A-Z]{1,2}\d{6,9}\b", lambda m: mask_id_like(m.group(0)), text)
    # Bank accounts (10-18 continuous digits)
    text = re.sub(r"\b\d{10,18}\b", lambda m: mask_id_like(m.group(0)), text)
    
    # 10. Addresses
    text = _mask_address_en_zh(text)

    return text

# ===========================
# Core Anonymization 
# ===========================
def desensitize_text(raw_text: str, ui_lang_code: str) -> str:
    if not raw_text.strip(): return ""

    analyzer = get_presidio_analyzer()
    lang = "zh" if re.search(r"[\u4e00-\u9fff]", raw_text[:500]) and ui_lang_code == "auto" else (ui_lang_code if ui_lang_code != "auto" else "en")

    # 1. Strong regex pre-replacement: Lock formatted PII
    processed_text = _regex_pre_pass(raw_text)

    # 2. Line-by-line NLP analysis: Prevent cross-line hallucination
    lines = processed_text.split('\n')
    out_lines =[]

    for line in lines:
        if not line.strip():
            out_lines.append(line)
            continue
            
        results: List[RecognizerResult] = analyzer.analyze(text=line, language=lang, entities=None)

        if not results:
            out_lines.append(line)
            continue

        results = sorted(results, key=lambda r: r.start)
        output =[]
        cursor = 0
        
        for res in results:
            start, end = res.start, res.end
            if start < cursor: continue
            output.append(line[cursor:start])
            
            entity_value = line[start:end]

            if re.fullmatch(r"\[REDACTED_[A-Z0-9_]+\]", entity_value) or "*" in entity_value:
                output.append(entity_value)
            else:
                masked_value = apply_custom_mask(res.entity_type, entity_value)
                output.append(masked_value)
            
            cursor = end
            
        output.append(line[cursor:])
        out_lines.append("".join(output))

    return '\n'.join(out_lines)


# ===========================
# Multi-file upload processing helpers
# ===========================
def _uploaded_files_signature(uploaded_files) -> str:
    """
    Build a stable signature for the current uploaded files list.
    Used to prevent repeated auto-processing on Streamlit reruns.
    """
    if not uploaded_files:
        return ""
    parts = []
    for f in uploaded_files:
        try:
            data = f.getvalue()
        except Exception:
            data = b""
        h = hashlib.sha1(data).hexdigest() if data else "0"
        parts.append(f"{getattr(f, 'name', '')}:{getattr(f, 'size', '')}:{h}")
    return "|".join(parts)


def _concat_with_headers(items: List[dict], text_key: str) -> str:
    blocks = []
    for it in items:
        name = it.get("name") or "uploaded"
        content = (it.get(text_key) or "").strip()
        if not content:
            continue
        blocks.append(f"===== FILE: {name} =====\n{content}")
    return "\n\n".join(blocks).strip()


# ===========================
# OpenAI compatible API call (Right-side API Geek Mode)
# ===========================

def stream_openai_chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    redacted_text: str,
) -> str:
    if not base_url:
        raise ValueError("Base URL cannot be empty.")
    if not api_key:
        raise ValueError("API Key cannot be empty.")
    if not redacted_text.strip():
        raise ValueError("Please anonymize the text on the left before sending it to the AI.")

    base = base_url.rstrip("/")
    if base.endswith("/v1/chat/completions") or base.endswith("/chat/completions"):
        url = base
    elif base.endswith("/v1"):
        url = base + "/chat/completions"
    else:
        url = base + "/v1/chat/completions"

    system_prompt = (
        "You are a privacy-preserving AI assistant. The user's content has already "
        "been locally anonymized. Only use the input text for your reasoning and "
        "never attempt to reconstruct any private information."
    )
    user_prompt = f"{prompt.strip()}\n\n--- Redacted text ---\n{redacted_text}"

    payload = {
        "model": model,
        "stream": True,
        "messages":[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        stream=True,
        timeout=60,
    )

    if response.status_code != 200:
        try:
            err_json = response.json()
            err_msg = err_json.get("error", {}).get("message", "")
        except Exception:
            err_msg = response.text[:300]
        raise RuntimeError(f"Request failed: HTTP {response.status_code} - {err_msg}")

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            data_str = line[len("data: ") :].strip()
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
                delta = data["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta
            except Exception:
                continue
    return

# ===========================
# Streamlit UI layout & interactions
# ===========================

st.set_page_config(page_title="GoCalma: Zero-Privacy-Leak AI Data Shield", layout="wide")

st.markdown(
    """
    <style>
    .gocalma-title {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        color: transparent;
        margin-bottom: 0.2rem;
    }
    .gocalma-subtitle {
        font-size: 0.95rem;
        color: #808897;
        margin-bottom: 1.2rem;
    }
    .gocalma-card {
        border-radius: 12px;
        padding: 1rem 1.2rem;
        background: linear-gradient(135deg, rgba(0, 201, 255, 0.12), rgba(146, 254, 157, 0.05));
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="gocalma-card">
      <div class="gocalma-title">GoCalma · Zero-Privacy-Leak AI Data Shield</div>
      <div class="gocalma-subtitle">
        Run a local anonymization shield before any data leaves your machine. No cloud pre-processing, no logs uploaded.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "redacted_text" not in st.session_state:
    st.session_state.redacted_text = ""
if "redacted_files" not in st.session_state:
    # List[{"name": str, "raw_text": str, "redacted_text": str}]
    st.session_state.redacted_files = []
if "uploaded_process_sig" not in st.session_state:
    st.session_state.uploaded_process_sig = ""
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""
if "api_response" not in st.session_state:
    st.session_state.api_response = ""
if "input_mode" not in st.session_state:
    st.session_state.input_mode = "文件"

col_left, col_mid, col_right = st.columns([1.1, 1.1, 1.1])

with col_left:
    st.markdown("### Input raw data with PII")
    lang_choice = st.selectbox("Select text language", LANG_OPTIONS, index=2, key="lang_choice")

    input_mode = st.radio(
        "Select input method",
        ["File", "Text"],
        horizontal=True,
        key="input_mode",
    )

    if input_mode == "File":
        uploaded_files = st.file_uploader(
            "Drag and drop files to upload (multiple files allowed, click × to remove)",
            type=["txt", "md", "json", "csv", "pdf"],
            accept_multiple_files=True,
            key="upload_file",
        )

        current_files_sig = _uploaded_files_signature(uploaded_files)
        lang_code_for_files = LANG_TO_CODE.get(st.session_state.lang_choice, "auto")
        process_sig = f"{current_files_sig}||{lang_code_for_files}"

        if process_sig != st.session_state.uploaded_process_sig:
            st.session_state.uploaded_process_sig = process_sig
            st.session_state.redacted_files = []
            st.session_state.redacted_text = ""

            if uploaded_files:
                extracted_items = []
                any_error = False

                for f in uploaded_files:
                    try:
                        extracted = extract_text_from_uploaded_file(f)
                    except Exception as e:
                        any_error = True
                        extracted = ""
                        st.error(f"Failed to read file: {getattr(f, 'name', 'unknown')} - {e}")

                    extracted_items.append({"name": getattr(f, "name", "uploaded"), "raw_text": extracted})

                redacted_items = []
                for it in extracted_items:
                    raw = (it.get("raw_text") or "").strip()
                    name = it.get("name") or "uploaded"
                    if not raw:
                        red = ""
                    else:
                        try:
                            red = desensitize_text(raw, lang_code_for_files)
                        except Exception as e:
                            any_error = True
                            red = ""
                            st.error(f"Desensitization failed: {name} - {e}")
                    redacted_items.append({"name": name, "raw_text": raw, "redacted_text": red})

                st.session_state.redacted_files = redacted_items
                combined = _concat_with_headers(redacted_items, "redacted_text")
                if combined.strip() or any_error:
                    st.session_state.redacted_text = combined

    else:
        raw_text = st.text_area(
            "Paste/enter text here:",
            height=260,
            value=st.session_state.raw_text,
            key="raw_text_area",
        )
        st.session_state.raw_text = raw_text

        btn_row1, btn_row2 = st.columns([1, 1])
        with btn_row1:
            do_desensitize = st.button("🛡️ Local Desensitization", use_container_width=True)
        with btn_row2:
            clear_input = st.button("🧹 Clear Input", use_container_width=True)

        if clear_input:
            st.session_state.raw_text = ""
            st.session_state.redacted_text = ""
            st.rerun()

        if do_desensitize:
            if not raw_text.strip():
                st.warning("Please enter text to desensitize.")
            else:
                with st.spinner("Processing local desensitization (offline spaCy model)..."):
                    try:
                        lang_code = LANG_TO_CODE[st.session_state.lang_choice]
                        redacted = desensitize_text(raw_text, lang_code)
                        st.session_state.redacted_text = redacted
                        st.success("Desensitization completed.")
                    except Exception as e:
                        st.error(f"Desensitization failed: {e}")

with col_mid:
    st.markdown("### Anonymized safe text")

    current_output = st.session_state.redacted_text or ""
    st.text_area("Desensitized Output (Read-only)", value=current_output, height=260, disabled=True)

    # If in file mode, show download list for output files
    if st.session_state.input_mode == "File" and st.session_state.redacted_files:
        st.markdown("#### Output File List (Download desensitized content)")
        for i, it in enumerate(st.session_state.redacted_files):
            name = it.get("name") or "uploaded"
            red = it.get("redacted_text") or ""
            if not red.strip():
                continue
            safe_name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
            st.download_button(
                label=f"Download: {safe_name}.redacted.txt",
                data=red.encode("utf-8"),
                file_name=f"{safe_name}.redacted.txt",
                mime="text/plain; charset=utf-8",
                use_container_width=True,
                key=f"dl_{i}_{safe_name}",
            )

        if current_output.strip():
            st.download_button(
                label="Download: All Files Combined (combined.redacted.txt)",
                data=current_output.encode("utf-8"),
                file_name="combined.redacted.txt",
                mime="text/plain; charset=utf-8",
                use_container_width=True,
                key="dl_combined",
            )

    # Copy always copies the *current* output used by the right-side AI.
    if current_output.strip():
        copy_html = f"""
        <div style="display:flex;align-items:center;gap:10px;">
          <button id="gocalma_copy_btn"
            style="background:#1f6feb;color:#fff;border:none;border-radius:6px;padding:6px 12px;cursor:pointer;font-size:0.85rem;">
            📋 Copy Output
          </button>
          <span id="gocalma_copy_msg" style="color:#808897;font-size:0.85rem;"></span>
          <script>
            const text = {json.dumps(current_output)};
            const btn = document.getElementById("gocalma_copy_btn");
            const msg = document.getElementById("gocalma_copy_msg");
            btn.addEventListener("click", async () => {{
              try {{
                await navigator.clipboard.writeText(text || "");
                msg.textContent = "Copied";
                setTimeout(() => msg.textContent = "", 1200);
              }} catch (e) {{
                msg.textContent = "Copy failed";
              }}
            }});
          </script>
        </div>
        """
        components.html(copy_html, height=55)

with col_right:
    st.markdown("### AI interaction (only sees anonymized text)")
    tab_api, tab_web = st.tabs(["API geek mode", "Web beginner mode"])

    with tab_api:
        base_url = st.text_input("Base URL", value="", placeholder="https://api.openai.com")
        api_key = st.text_input("API Key", type="password", value="")
        model_name = st.text_input("Model name", value="gpt-4o-mini")
        custom_prompt = st.text_area("Custom prompt", value="Summarize this safely.", height=120, key="custom_prompt_area")

        send_to_ai = st.button("🚀 Send to AI", use_container_width=True)

        if send_to_ai:
            if not st.session_state.redacted_text.strip():
                st.warning("Please anonymize the text first.")
            else:
                with st.spinner("Calling API..."):
                    placeholder = st.empty()
                    accumulated = ""
                    try:
                        for delta in stream_openai_chat_completion(
                            base_url.strip(), api_key.strip(), model_name.strip(), custom_prompt, st.session_state.redacted_text
                        ):
                            accumulated += delta
                            placeholder.markdown(accumulated)
                    except Exception as e:
                        st.error(f"Error: {e}")
                    st.session_state.api_response = accumulated

        if not send_to_ai and st.session_state.api_response:
            st.markdown("#### Most recent AI response")
            st.markdown(st.session_state.api_response)

    with tab_web:
        st.markdown("**No API key? Just copy from the middle and paste to the web:**")
        c1, c2, c3 = st.columns(3)
        with c1: st.link_button("👉 ChatGPT", "https://chat.openai.com", use_container_width=True)
        with c2: st.link_button("👉 Claude", "https://claude.ai", use_container_width=True)
        with c3: st.link_button("👉 Kimi", "https://kimi.moonshot.cn", use_container_width=True)