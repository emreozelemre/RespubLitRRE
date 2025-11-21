import os
import re
import json
from typing import Optional, Dict, Any, List

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import gradio as gr

try:
    import pypdf
except ImportError:
    pypdf = None
try:
    import docx
except ImportError:
    docx = None

# --------------------------
# Config
# --------------------------
# Base Mistral model on HF (unchanged, public)
BASE_MODEL_ID = os.environ.get(
    "RESPUBLIT_BASE_MODEL_ID",
    "mistralai/Mistral-7B-Instruct-v0.3",
)

# LoRA adapter on HF for RespubLitRRE (LoRA-only repo)
LORA_ADAPTER_ID = os.environ.get(
    "RESPUBLIT_LORA_ADAPTER_ID",
    "emreozelemre/RespubLitRRE-LoRA",
)

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
DEFAULT_LOGO_PATH = os.path.join(ASSETS_DIR, "respublit_logo.jpg")

LOGO_PATH = os.environ.get("RESPUBLIT_LOGO_PATH", DEFAULT_LOGO_PATH)

# Context / length limits
S2_MAX_INPUT_TOKENS = 8192
S2_MAX_NEW_TOKENS = 384

# Character-level caps before tokenization
ABSTRACT_CHAR_LIMIT = 1200
REVIEW_CHAR_LIMIT = 12000 

# --------------------------
# File helpers
# --------------------------
def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_pdf(path: str) -> str:
    if not pypdf:
        return ""
    try:
        reader = pypdf.PdfReader(path)
        out = []
        for p in reader.pages:
            t = p.extract_text() or ""
            if t.strip():
                out.append(t)
        return "\n".join(out)
    except Exception:
        return ""


def _read_docx(path: str) -> str:
    if not docx:
        return ""
    try:
        d = docx.Document(path)
        return "\n".join(p.text for p in d.paragraphs)
    except Exception:
        return ""


def _load_file_to_text(file_obj) -> str:
    if file_obj is None:
        return ""
    path = file_obj.name
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return _read_txt(path)
    if ext == ".pdf":
        return _read_pdf(path)
    if ext == ".docx":
        return _read_docx(path)
    return ""


def _truncate(text: str, n: int) -> str:
    text = text or ""
    return text[:n] if len(text) > n else text


# --------------------------
# JSON extraction
# --------------------------
def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Robust JSON extractor:
    - find the first '{'
    - walk backwards from the end for '}' candidates
    - try json.loads on each decreasing prefix
    """
    if not text or len(text.strip()) < 5:
        return None

    text = text.strip()
    start = text.find("{")
    if start == -1:
        return None

    s = text[start:]
    last_brace_positions = [i for i, ch in enumerate(s) if ch == "}"]
    if not last_brace_positions:
        return None

    for end in reversed(last_brace_positions):
        candidate = s[: end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            continue

    # last resort: strip trailing commas
    candidate = re.sub(r",(\s*[}\]])", r"\1", s)
    try:
        return json.loads(candidate)
    except Exception:
        return None


# --------------------------
# Convert JSON → UI pieces
# --------------------------
def _build_ui_from_json(obj: Dict[str, Any]) -> (List[Dict[str, Any]], str, str):
    """
    Returns:
      - scores: list of dicts [{Dimension, Score}]
      - justification_md: short markdown
      - bias_md: short markdown
    """
    eval_block = obj.get("evaluation", {}) or {}
    bias_block = obj.get("bias_signals", {}) or {}

    tr = eval_block.get("technical_rigor")
    cf = eval_block.get("constructive_feedback")
    oq = eval_block.get("overall_quality")
    rel = eval_block.get("relevance_to_abstract")
    just = (eval_block.get("justification") or "").strip()

    sb_flag = bias_block.get("sentiment_bias_flag", None)
    misaligned = bias_block.get("misaligned_count", None)
    avg_align = bias_block.get("avg_sentiment_alignment", None)
    bias_prob = bias_block.get("bias_probability", None)
    bias_sev = bias_block.get("bias_severity", None)

    # Short labels: TR, CF, OQ, RA
    scores: List[Dict[str, Any]] = []
    if isinstance(tr, (int, float)):
        scores.append({"Dimension": "TR", "Score": tr})
    if isinstance(cf, (int, float)):
        scores.append({"Dimension": "CF", "Score": cf})
    if isinstance(oq, (int, float)):
        scores.append({"Dimension": "OQ", "Score": oq})
    if isinstance(rel, (int, float)):
        scores.append({"Dimension": "RA", "Score": rel})

    # Collapsed bias label
    bias_high = False
    if isinstance(bias_sev, str) and bias_sev in {"High", "Moderate"}:
        bias_high = True
    elif isinstance(bias_prob, (int, float)) and bias_prob is not None and bias_prob >= 0.4:
        bias_high = True
    bias_level = "High / Moderate" if bias_high else "None / Low"

    justification_md = just if just else "_No justification returned by the model._"

    lines = []
    lines.append(f"**Bias level (collapsed):** {bias_level}")
    if sb_flag is not None:
        lines.append(f"**Sentiment bias flag:** `{bool(sb_flag)}`")
    if isinstance(misaligned, int):
        lines.append(f"**Text–score misalignment count:** `{misaligned}`")
    if isinstance(avg_align, (int, float)):
        lines.append(f"**Average sentiment alignment:** `{avg_align:.3f}`")
    if isinstance(bias_prob, (int, float)):
        lines.append(f"**Bias probability (model estimate):** `{bias_prob:.3f}`")
    if isinstance(bias_sev, str) and bias_sev:
        lines.append(f"**Bias severity (model estimate):** `{bias_sev}`")

    bias_md = "\n\n".join(lines) if lines else "_No bias information returned._"

    return scores, justification_md, bias_md


# --------------------------
# Model loading (8-bit) – RespubLitRRE
# --------------------------
def load_model_and_tokenizer():
    """
    Load base Mistral-7B-Instruct and apply the RespubLitRRE LoRA adapter.

    Defaults:
    - BASE_MODEL_ID: mistralai/Mistral-7B-Instruct-v0.3
    - LORA_ADAPTER_ID: emreozelemre/RespubLitRRE-LoRA

    Both can be overridden via environment variables:
    - RESPUBLIT_BASE_MODEL_ID
    - RESPUBLIT_LORA_ADAPTER_ID

    The base model is loaded in 8-bit using BitsAndBytesConfig(load_in_8bit=True),
    which assumes a GPU with at least 16 GB of VRAM for reliable inference.
    """
    bnb_cfg = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    print(f"[load] Loading base model from: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_cfg,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    model = base_model
    if LORA_ADAPTER_ID:
        try:
            print(f"[load] Loading RespubLitRRE LoRA adapter from: {LORA_ADAPTER_ID}")
            # Works for both HF repos and local directories
            model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_ID)
            print("[load] LoRA adapter successfully loaded.")
        except Exception as e:
            print(f"[load] WARNING: could not load LoRA adapter '{LORA_ADAPTER_ID}': {e}")
            print("[load] Proceeding with base model only (results will differ).")

    model.eval()
    model.config.use_cache = True
    return model, tokenizer

# load once globally (no re-loading on each click)
MODEL, TOKENIZER = load_model_and_tokenizer()


# --------------------------
# SHORT, ROBUST PROMPT (RespubLit Review Report Evaluator / RespubLitRRE)
# --------------------------
S2_SYSTEM_MSG = (
    "You evaluate the QUALITY of peer-review reports, not the project itself.\n"
    "- Input: optional abstract + one review text.\n"
    "- Output: a single JSON object with keys 'evaluation' and 'bias_signals'.\n\n"

    "GENERAL PRINCIPLES:\n"
    "- You judge HOW the reviewer reviews: clarity, structure, depth, specificity, reasoning, and usefulness.\n"
    "- You NEVER evaluate the proposal/paper itself.\n"
    "- Long conceptual or theoretical critiques count as high-quality analytical work.\n"
    "- Substance outweighs neat formatting. Engagement outweighs politeness.\n"
    "- Respond with JSON only.\n\n"

    "=== FALSE RIGOR WARNING (IMPORTANT FOR ACCURATE SCORING) ===\n"
    "- Do NOT reward reviews solely because they use complex technical terms, advanced jargon, or a confident tone.\n"
    "- Technical language WITHOUT clear explanation or WITHOUT linking the criticism to the proposal counts as LOWER rigor.\n"
    "- If the review lists \"methodological flaws\" but does NOT show:\n"
    "     (1) where this appears in the proposal, AND\n"
    "     (2) WHY it matters in practical terms,\n"
    "  then technical_rigor MUST NOT be above 3.\n"
    "- If a critique sounds complicated but is not clearly reasoned or grounded, score it LOWER, not higher.\n\n"

    "=== TECHNICAL_RIGOR (CONCEPTUAL OR METHODOLOGICAL) ===\n"
    "Technical rigor measures how deeply the review engages with the substance of the work. Rigor can come from:\n"
    "- methodological critique (design, assumptions, data, models), OR\n"
    "- conceptual depth (definitions, theory, framework logic, argument structure), OR\n"
    "- structural analysis (coherence, layer logic, internal consistency), OR\n"
    "- literature-informed critique (gaps, theoretical misalignment, missing dimensions).\n\n"

    "SCORING:\n"
    "- 5 = Exceptional rigor: long, detailed, highly analytical review with multiple precise and clearly explained points.\n"
    "- 4 = Strong rigor: many substantial points with clear reasoning.\n"
    "- 3 = Moderate rigor: relevant, but explanations or grounding may be uneven.\n"
    "- 2 = Low rigor: SOME real engagement, but shallow.\n"
    "- 1 = Minimal rigor: vague praise, empty adjectives, no real reasoning.\n\n"

    "SPECIAL RULE FOR EMPTY PRAISE:\n"
    "- Reviews that say things like 'excellent', 'great', 'good', 'robust', etc. WITHOUT justification "
    "must receive technical_rigor = 1 unless they contain additional reasoning.\n\n"

    "=== CONSTRUCTIVE_FEEDBACK ===\n"
    "Constructive feedback measures how useful the review is for improving the work. Actionable feedback includes:\n"
    "- specific methodological or conceptual suggestions,\n"
    "- identifying missing elements AND explaining why they matter,\n"
    "- proposing better structure, definitions, frameworks, or clarifications.\n\n"

    "SCORING:\n"
    "- 5 = Many concrete, well-targeted, actionable suggestions.\n"
    "- 4 = Several clear and useful suggestions.\n"
    "- 3 = Some suggestions but uneven.\n"
    "- 2 = At least ONE concrete suggestion with explanation.\n"
    "- 1 = No actionable advice; only praise or vague criticism.\n\n"

    "=== OVERALL_QUALITY ===\n"
    "Overall quality evaluates structure, clarity, coherence, depth, and usefulness.\n\n"

    "SCORING:\n"
    "- 5 = Excellent: structured, coherent, rich in insight.\n"
    "- 4 = Strong: detailed and well-organized.\n"
    "- 3 = Adequate: useful but uneven or partly superficial.\n"
    "- 2 = Weak: limited depth but contains SOME substance.\n"
    "- 1 = Poor: shallow, generic, unhelpful.\n\n"

    "=== DOMINANT SUBSTANCE RULE ===\n"
    "- If the review identifies MORE THAN ONE substantive weakness (methodological, conceptual, structural), "
    "even if messy or disorganized, then:\n"
    "    * technical_rigor MUST be >= 2\n"
    "    * constructive_feedback MUST be >= 2 IF the reviewer explains WHY the weakness matters\n"
    "    * overall_quality MUST be >= 2\n\n"

    "=== RELEVANCE_TO_ABSTRACT ===\n"
    "- If no abstract is provided: relevance_to_abstract = null.\n"
    "- Otherwise score how well the review engages with the abstract's main aims.\n\n"

    "=== META-RULES ===\n"
    "- Do NOT evaluate the project/paper/proposal.\n"
    "- Judge ONLY the quality of the REVIEW.\n"
    "- Avoid defaulting to '3'.\n"
    "- Substance > tone. Reasoning > jargon.\n"
    "- Use '5' ONLY for exceptional, well-explained, well-grounded reviews.\n"
)

def build_stage2_messages(abstract: Optional[str], review_text: str):
    """
    Build chat messages for stage-2 evaluation.
    Abstract is clearly scoped to relevance_to_abstract.
    Review text drives TR/CF/OQ.
    """
    abstract = (abstract or "").strip()
    review_text = (review_text or "").strip()

    # compute word count here (fix NameError)
    review_word_count = len(review_text.split()) if review_text else 0

    if abstract:
        header_txt = (
            "ABSTRACT (use ONLY for relevance_to_abstract):\n"
            f"{abstract}\n\n"
        )
        ra_line = (
            "An abstract is provided. Set relevance_to_abstract to an integer 1–5 based on how well the "
            "review engages with the main aims/questions/methods described in the abstract.\n"
        )
    else:
        header_txt = "NO ABSTRACT PROVIDED. You must set relevance_to_abstract to null.\n\n"
        ra_line = "No abstract is provided. relevance_to_abstract must be null.\n"

    meta_line = (
        f"META: This review is approximately {review_word_count} words long. "
        "If it is long and contains many specific points, technical_rigor and constructive_feedback "
        "should not be below 3 unless the text is incoherent.\n"
    )

    user_msg = (
        header_txt +
        "REVIEW TEXT (judge only this for technical_rigor, constructive_feedback, overall_quality):\n"
        f"{review_text}\n\n"
        f"{ra_line}\n"
        f"{meta_line}\n"
        "JUSTIFICATION REQUIREMENTS:\n"
        "- In the justification, you must NOT directly criticise or evaluate the paper/project.\n"
        "- You MUST talk about the REVIEW itself, for example:\n"
        "  * how specific it is (does it cite sections/pages/references?),\n"
        "  * how many concrete suggestions it gives,\n"
        "  * how well structured and clear it is,\n"
        "  * whether its criticisms are explained or just asserted.\n"
        "- Avoid long restatements of the paper's weaknesses. Focus instead on whether the REVIEW explains "
        "those weaknesses with enough detail, examples, and structure.\n"
        "- You may say 'the review points out X' or 'the review mentions weaknesses in Y', "
        "but you must NOT present those weaknesses as your own assessment of the paper.\n\n"
        "Now output a single JSON object with this structure:\n"
        "{\n"
        '  "evaluation": {\n'
        '    "technical_rigor": <int 1-5>,\n'
        '    "constructive_feedback": <int 1-5>,\n'
        '    "overall_quality": <int 1-5>,\n'
        '    "relevance_to_abstract": <int 1-5 or null>,\n'
        '    "justification": "<max 60 words describing the quality of the REVIEW (not the paper)>"\n'
        "  },\n"
        '  "bias_signals": {\n'
        '    "sentiment_bias_flag": <true/false>,\n'
        '    "misaligned_count": <int>,\n'
        '    "avg_sentiment_alignment": <float or null>,\n'
        '    "bias_probability": <float 0-1 or null>,\n'
        '    "bias_severity": "None" | "Low" | "Moderate" | "High" | null,\n'
        '    "per_question": []\n'
        "  }\n"
        "}\n\n"
        "Output ONLY this JSON, with no explanations."
    )

    messages = [
        {"role": "system", "content": S2_SYSTEM_MSG},
        {"role": "user",  "content": user_msg},
    ]
    return messages

# --------------------------
# Core Stage-2 evaluation
# --------------------------
@torch.inference_mode()
def evaluate_review_stage2(proposal_file, review_file, proposal_text, review_text):
    prop_file = _load_file_to_text(proposal_file) if proposal_file else ""
    rev_file = _load_file_to_text(review_file) if review_file else ""

    abstract = _truncate(prop_file.strip() or (proposal_text or "").strip(), ABSTRACT_CHAR_LIMIT)
    review = _truncate(rev_file.strip() or (review_text or "").strip(), REVIEW_CHAR_LIMIT)

    empty_df = pd.DataFrame(columns=["Dimension", "Score"])

    if not review:
        return empty_df, "Please provide a review report.", ""

    model = MODEL
    tokenizer = TOKENIZER

    messages = build_stage2_messages(abstract if abstract else None, review)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=S2_MAX_INPUT_TOKENS,
    ).to(model.device)

    print(f"[diag] Stage2 input token length = {inputs['input_ids'].shape[1]}")

    out = model.generate(
        **inputs,
        max_new_tokens=S2_MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=False,
    )

    gen_ids = out[0][inputs["input_ids"].shape[1] :]
    raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    parsed = _extract_json_from_text(raw_text)

    if not parsed:
        return (
            empty_df,
            "⚠️ The model did not return a clearly parseable Stage-2 JSON object.\n\n"
            "Raw model output:\n\n```text\n" + raw_text + "\n```",
            "",
        )

    scores_list, just_md, bias_md = _build_ui_from_json(parsed)
    if scores_list:
        df = pd.DataFrame(scores_list)
    else:
        df = empty_df

    return df, just_md, bias_md


# --------------------------
# UI – RespubLit Review Report Evaluator (RespubLitRRE)
# --------------------------
def launch_ui():
    css = """
    body { background-color: #0f172a; }
    .rep-header-row { align-items: center; margin-bottom: 1.0rem; }
    .rep-logo img { max-height: 80px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.25); }
    .rep-title { font-size: 1.6rem !important; font-weight: 700 !important; color: #e5e7eb !important; }
    .rep-subtitle { font-size: 0.9rem !important; color: #9ca3af !important; }
    .rep-card { background: #111827; border-radius: 18px; padding: 16px 18px; box-shadow: 0 10px 25px rgba(0,0,0,0.35); border: 1px solid rgba(148,163,184,0.25); }
    .rep-label { font-size: 0.85rem; color: #9ca3af; margin-bottom: 0.5rem; }
    """

    try:
        from gradio.themes import Soft
        theme = Soft(primary_hue="indigo", neutral_hue="slate")
    except Exception:
        theme = gr.themes.Soft()

    with gr.Blocks(
        theme=theme,
        title="RespubLit Review Report Evaluator (RespubLitRRE)",
        css=css,
    ) as demo:
        # Header
        with gr.Row(elem_classes="rep-header-row"):
            with gr.Column(scale=1, min_width=80):
                if os.path.exists(LOGO_PATH):
                    gr.Image(
                        value=LOGO_PATH,
                        show_label=False,
                        elem_classes="rep-logo",
                        interactive=False,
                        show_download_button=False,
                        show_fullscreen_button=False,
                    )
                else:
                    gr.Markdown(" ")
            with gr.Column(scale=5):
                gr.Markdown(
                    "**RespubLit Review Report Evaluator (RespubLitRRE)**",
                    elem_classes="rep-title",
                )
                gr.Markdown(
                    "AI-assisted evaluation of peer review reports and potential bias signals.",
                    elem_classes="rep-subtitle",
                )

        with gr.Row():
            with gr.Column(scale=1, elem_classes="rep-card"):
                gr.Markdown("**Input**", elem_classes="rep-label")
                with gr.Row():
                    with gr.Column():
                        proposal_file = gr.File(
                            label="Abstract (.pdf/.docx/.txt, optional)",
                            file_types=[".pdf", ".docx", ".txt"],
                        )
                        with gr.Accordion(
                            "Paste abstract text (optional)", open=False
                        ):
                            proposal_text = gr.Textbox(
                                label=None,
                                lines=4,
                                placeholder="Paste abstract text here (optional)",
                            )
                    with gr.Column():
                        review_file = gr.File(
                            label="Review report (.pdf/.docx/.txt)",
                            file_types=[".pdf", ".docx", ".txt"],
                        )
                        with gr.Accordion("Paste review text", open=False):
                            review_text = gr.Textbox(
                                label=None,
                                lines=8,
                                placeholder="Paste review text here",
                            )

                run_btn = gr.Button("Evaluate review", variant="primary")

        with gr.Row():
            with gr.Column(scale=1, elem_classes="rep-card"):
                gr.Markdown("**Score breakdown**", elem_classes="rep-label")
                score_plot = gr.BarPlot(
                    x="Dimension",
                    y="Score",
                    x_title="",
                    y_title="",
                    y_lim=(0, 5),
                    show_label=False,
                    height=260,
                )
                gr.Markdown(
                    "_TR = Technical rigor · CF = Constructive feedback · "
                    "OQ = Overall quality · RA = Relevance to abstract_",
                    elem_classes="rep-label",
                )

            with gr.Column(scale=1, elem_classes="rep-card"):
                gr.Markdown("**Justification**", elem_classes="rep-label")
                justification_md = gr.Markdown()

            with gr.Column(scale=1, elem_classes="rep-card"):
                with gr.Accordion("Bias signals", open=True):
                    bias_md = gr.Markdown()

        run_btn.click(
            fn=evaluate_review_stage2,
            inputs=[proposal_file, review_file, proposal_text, review_text],
            outputs=[score_plot, justification_md, bias_md],
        )

    return demo


if __name__ == "__main__":
    demo = launch_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)

