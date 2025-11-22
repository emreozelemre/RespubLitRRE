# RespubLit Review Report Evaluator (RespubLitRRE)

Official Gradio application and inference pipeline for the RespubLit Review Report Evaluator (RespubLitRRE).

RespubLitRRE is a specialised language model (Mistral-7B + LoRA) designed to assess the **quality of peer review reports**.

It evaluates:

- **Technical Rigor (TR)**
- **Constructive Feedback (CF)**
- **Overall Quality (OQ)**
- **Relevance to Abstract (RA)**
- **Bias Signals** (sentiment bias, misalignment, etc.)

The model **evaluates the review itself**, not the underlying project or proposal.

This repo provides the **calibrated Gradio app** that wraps the model with:
- carefully tuned prompts,
- robust JSON extraction and parsing,
- length control and truncation,
- a visual score breakdown,
- and a bias-signal summary.

⚠️ For best and most stable results, users are strongly encouraged to run this calibrated Gradio application rather than calling the LoRA adapter directly. The full pipeline (model loading, prompting, parsing, inconsistency handling, and scoring) is implemented in this app.

⚠️ The model requires at least 16GB of VRAM for reliable inference. Lower VRAM configurations may fail or silently truncate inputs.

---

## 🔗 Related Hugging Face resources

- **LoRA adapter:** [`emreozelemre/RespubLitRRE-LoRA`](https://huggingface.co/emreozelemre/RespubLitRRE-LoRA) 
- **Base model:** [`mistralai/Mistral-7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)

This Gradio app automatically loads:

- the base model from `mistralai/Mistral-7B-Instruct-v0.3`
- the LoRA adapter from `emreozelemre/RespubLitRRE-LoRA`

using 8-bit quantization by default (via `BitsAndBytesConfig(load_in_8bit=True)`).

---

## 🧰 What’s in this repo?

- `RespubLitRRE_Gradio.py` — main app and pipeline (model loading, prompts, JSON parsing, UI) 
- `requirements.txt` — Python dependencies 
- `LICENSE` — Non-Commercial OpenRAIL-M licence (same as the model) 
- `README.md` — this file
- `Installation Instructions.pdf` — Comprehensive cross-platform installation guide 

You can treat `RespubLitRRE_Gradio.py` as the **canonical implementation**
of the RespubLitRRE pipeline.

**This app will:**

Load the base Mistral model in 8-bit,

Load the RespubLitRRE LoRA from Hugging Face,

Launch a Gradio interface at http://0.0.0.0:7860 (or http://localhost:7860).


**If you need to override the default IDs, you can set:**

export RESPUBLIT_BASE_MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"

export RESPUBLIT_LORA_ADAPTER_ID="emreozelemre/RespubLitRRE-LoRA"


**Make sure you have a working GPU setup with recent torch and CUDA drivers if you want reasonable inference speed.**

## 🧠 How to use the app

**Abstract (optional)**

Upload a .pdf, .docx, or .txt file, or paste the abstract text.

**Review report (required)**

Upload the review (.pdf, .docx, .txt) or paste the text.

**Click “Evaluate review”.**

The app will display:

A bar plot with TR / CF / OQ / RA scores,

A short justification (focusing on the quality of the review),

A bias signals panel (collapsed bias level, misalignment count, etc.).

**All of this logic is implemented in RespubLitRRE_Gradio.py.**

## ⚖️ License
The code and pipeline in this repository follow the same Non-Commercial OpenRAIL-M–style licence as the LoRA adapter:

Free for research, academic, and personal use.

Commercial use requires a separate licence from RespubLit.

For commercial licensing, contact:

**RespubLit**

Website:    **https://www.respublit.org**

Email:      **emreozel@respublit.org**

## 📄 Citation
If you use RespubLitRRE in research, please cite:

**Özel, E. (2025). RespubLit Review Report Evaluator (RespubLitRRE) — LoRA Adapter and Gradio Application. Hugging Face & GitHub.**

## 🛡 Ethical use
**Users agree not to misuse this tool, including:**

automated exclusion of reviewers,

manipulation of funding outcomes,

surveillance or profiling of reviewers,

harmful, deceptive, or discriminatory applications.

Always keep a human in the loop when using the outputs in evaluation or decision-making contexts.

