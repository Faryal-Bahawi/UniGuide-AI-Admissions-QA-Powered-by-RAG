# 🎓 UniGuide AI — Admissions Q&A Powered by RAG

> A Retrieval-Augmented Generation system that answers University of New Haven admissions questions using official university documents — not LLM guesswork.

---

## 📌 The Problem

Anyone who has tried to find specific admissions information on a university website knows how painful it can be. At UNH, details about deadlines, required documents, tuition, and international student policies are spread across dozens of separate pages. Students waste time clicking around, end up on the wrong page, or just email the admissions office asking something that's already published somewhere.

UniGuide AI fixes this by letting students ask plain questions and get accurate, document-grounded answers instantly.

---

## 💡 What This Project Does

UniGuide AI uses a RAG (Retrieval-Augmented Generation) pipeline:

1. It searches through a curated set of official UNH admissions documents
2. Pulls the most relevant chunks of text for a given question
3. Passes those chunks to a language model as context
4. Returns a grounded, one-sentence answer based only on what the documents say

No hallucination. No made-up deadlines. Just answers pulled from real sources.

---

## 🏗️ System Architecture

```
User Question
     ↓
[ Encode query with all-MiniLM-L6-v2 ]
     ↓
[ Cosine Similarity Search over 220 document chunks ]
     ↓
[ Keyword Boosting + Source Penalties ]
     ↓
[ Top 5 Relevant Chunks Retrieved ]
     ↓
[ Structured Prompt + Context → Gemma-2b-it ]
     ↓
Grounded Answer
```

---

## 📂 Project Structure

```
uniguide-ai/
│
├── NLP_Project.ipynb               # Full pipeline notebook
├── README.md                       # You are here
├── UniGuide_AI_Paper.docx          # Course project report (ACL-style)
│
└── data/
    ├── unh_deadlines.txt           # Application deadlines
    ├── unh_requirements_grad.txt   # Graduate admissions requirements
    ├── unh_requirements_undergrad.txt  # Undergraduate requirements
    ├── unh_international.txt       # International student policies
    ├── unh_tuition.txt             # Tuition & cost of attendance
    ├── unh_faq.txt                 # General admissions FAQ
    └── unh_programs.txt            # Selected program descriptions
```

---

## 🛠️ Tech Stack

| Component | Tool / Model |
|---|---|
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Similarity Search | Cosine similarity via scikit-learn |
| Main Generator | `Gemma-2b-it` (Google DeepMind, HuggingFace) |
| Baseline Generator | `DistilGPT2` (HuggingFace) |
| Framework | Python, PyTorch, Transformers |
| Environment | Google Colab (GPU recommended) |

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/uniguide-ai.git
cd uniguide-ai
```

### 2. Install dependencies

```bash
pip install sentence-transformers transformers accelerate scikit-learn torch huggingface_hub
```

### 3. Authenticate with Hugging Face

Gemma-2b-it is a gated model. You need a free Hugging Face account and must accept the model terms at:
👉 https://huggingface.co/google/gemma-2b-it

Then log in:

```python
from huggingface_hub import login
login()  # paste your HF token when prompted
```

---

## ▶️ How to Run

Open `NLP_Project.ipynb` in Google Colab or Jupyter and run all cells top to bottom. The notebook will:

- Load the 7 admissions documents from the `data/` folder
- Chunk and embed all text
- Load Gemma-2b-it and DistilGPT2
- Run 10 evaluation questions through both models
- Display and compare outputs

### Ask your own questions

After the setup cells run, you can query the system directly:

```python
# Main system (Gemma-2b-it)
print(rag_answer("What is the application deadline for international students?"))

# Baseline (DistilGPT2)
print(rag_answer_small("What is the application deadline for international students?"))
```

---

## 💬 Example Questions & Outputs

| Question | Gemma-2b-it Answer |
|---|---|
| What is the priority deadline for international graduate students? | The priority deadline for international graduate students is February 1 for Fall and October 1 for Spring admission. |
| Do graduate applicants need GRE or GMAT scores? | GRE or GMAT scores are not required for most graduate programs at UNH, though individual programs may differ. |
| What is required for first-year undergraduate applicants? | First-year applicants must submit a completed application, official high school transcripts, SAT/ACT scores, a personal essay, and two letters of recommendation. |

> DistilGPT2 on the same questions: repeated the question, looped, or generated off-topic text every single time.

---

## 📊 Evaluation Results

**10 domain-specific questions, manually scored 1–5 across accuracy, relevance, groundedness, and clarity.**

| Model | Avg Score | Questions Won |
|---|---|---|
| Gemma-2b-it | **3.75 / 5** | 10 / 10 |
| DistilGPT2 | 1.00 / 5 | 0 / 10 |

**Score breakdown — Gemma-2b-it:**

| Question | Score |
|---|---|
| Priority deadline for intl. grad students | 5.0 |
| Fall/spring deadlines for grad students | 4.5 |
| Documents for grad application | 4.0 |
| GRE/GMAT requirement for grad applicants | 4.0 |
| English test scores for intl. students | 4.0 |
| Documents for intl. grad applicants | 3.5 |
| Documents for transfer undergrad applicants | 4.0 |
| Requirements for first-year undergrad | 5.0 |
| Total cost of attendance on campus | 3.0 |
| Graduate tuition per credit hour | 3.5 |

---

## 🔍 Key Insights

**1. Instruction tuning is not optional.**
DistilGPT2 had access to the exact same retrieved context as Gemma but couldn't use it. Without instruction tuning, the model just continues text — it doesn't answer questions.

**2. Retrieval quality matters as much as the model.**
Early versions of the retriever kept pulling undergraduate chunks for graduate-specific questions because the vocabulary overlaps. Adding source-based boosting and keyword penalties fixed this without touching the model at all.

**3. RAG reduces hallucination significantly.**
Without retrieved context, neither model could answer UNH-specific questions reliably. With RAG, Gemma answered 8 out of 10 questions correctly.

**4. Chunking affects how complete the answers are.**
300-character chunks sometimes cut sentences mid-phrase. One question scored 3.0 instead of 5.0 purely because the dollar amount was in the next chunk. Sentence-aware chunking would fix this.

---

## ⚠️ Limitations

- Corpus only covers 7 document types — program-specific pages, scholarships, and housing are not included
- Character-level chunking occasionally produces incomplete context
- Evaluation is manual and single-annotator (no ROUGE or BERTScore yet)
- Gemma-2b-it requires a GPU for practical inference speed
- The retriever uses flat cosine search — slow at larger scale

---

## 🚀 Future Work

- [ ] Switch to sentence-aware chunking to fix mid-phrase cuts
- [ ] Add a FAISS index for faster retrieval at scale
- [ ] Evaluate with ROUGE-L and BERTScore for objective metrics
- [ ] Expand the corpus to cover more UNH programs and services
- [ ] Build a simple web UI (Gradio or React) for student-facing use
- [ ] Add a cross-encoder re-ranker between retrieval and generation

---

## 📄 Report

A full project report written in ACL conference style is included in the repo:
📎 `UniGuide_AI_Paper.docx`

It covers the full methodology, related work, results analysis, and references.

---

## 👩‍💻 Author

**Faryal Bahawi**
Department of Data Science — University of New Haven
NLP Course Project | Professor Khaled Sayed | May 2026

---

## 📚 References

- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.
- Reimers & Gurevych (2019). Sentence-BERT. *EMNLP*.
- Google DeepMind Team (2024). Gemma. *arXiv:2403.08295*.
- Sanh et al. (2019). DistilBERT. *arXiv:1910.01108*.
- Wei et al. (2022). Finetuned Language Models are Zero-Shot Learners. *ICLR*.
