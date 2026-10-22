# UniGuide AI: Admissions Questions, Answered from the Source

If you've ever tried to find one specific detail on a university website, like the deadline for international graduate students or whether you need the GRE, you know how scattered that information is. At the University of New Haven it's spread across dozens of pages, so students click around, land on the wrong page, or email the admissions office about something that's already published.

UniGuide AI lets you ask the question in plain English and get a short answer pulled straight from the official admissions documents, rather than whatever a language model happens to remember.

*NLP course project, University of New Haven, Spring 2026*

## How it works

```
Your question
   ↓  encoded with all-MiniLM-L6-v2 (Sentence-BERT)
Cosine-similarity search over 220 document chunks
   ↓  keyword boosting and source penalties
The 5 most relevant chunks
   ↓  passed to Gemma-2b-it with a structured prompt
A short answer grounded in the documents
```

The knowledge base is seven official admissions documents covering deadlines, graduate and undergraduate requirements, international student policies, tuition, an FAQ, and program descriptions. Together they come to about 14,000 tokens, split into 220 chunks.

## How well it works

I wrote 10 realistic admissions questions and scored every answer by hand on a 5-point rubric covering accuracy, relevance, grounding in the sources, and clarity.

| Model | Average score | Questions won |
|---|---|---|
| **Gemma-2b-it** | **3.75 / 5** | 10 of 10 |
| DistilGPT2 (baseline) | 1.00 / 5 | 0 of 10 |

A few things I learned along the way:

- **Instruction tuning matters more than I expected.** DistilGPT2 saw exactly the same retrieved passages as Gemma and still couldn't use them. It repeated the question, looped, or drifted off topic every time.
- **Retrieval is half the battle.** Early on, graduate questions kept pulling undergraduate passages because the wording overlaps so much. Adding keyword boosting and source penalties fixed this without changing the model at all, and the affected questions went from 2.0 to 5.0.
- **Chunking changes answers.** One question scored 3.0 instead of 5.0 only because the dollar amount it needed sat in the next chunk.

## What's in this repository

| Folder | Contents |
|---|---|
| `notebook/` | The full pipeline notebook |
| `data/` | The seven admissions documents (`data.zip`) |
| `report/` | The project report, written in ACL conference style |
| `slides/` | Presentation slides |

## Running it yourself

1. Unzip `data/data.zip`.
2. Install the libraries:
   ```bash
   pip install sentence-transformers transformers accelerate scikit-learn torch huggingface_hub
   ```
3. Gemma is a gated model, so accept its terms at [huggingface.co/google/gemma-2b-it](https://huggingface.co/google/gemma-2b-it) and log in with your Hugging Face token.
4. Open the notebook in Google Colab (a GPU helps) or Jupyter and run all cells.

After setup you can ask your own questions:

```python
print(rag_answer("What is the application deadline for international students?"))
```

## What I'd do next

Switch to sentence-aware chunking, add a FAISS index so retrieval scales, measure answers with ROUGE-L and BERTScore alongside my manual scores, and put a simple web interface on top so students could actually use it.

## Built with

Python, PyTorch, HuggingFace Transformers, sentence-transformers, scikit-learn and Google Colab.

---

Made by **Faryal Bahawi**. More of my work is on my [portfolio](https://faryal-bahawi.netlify.app), and I'm on [LinkedIn](https://www.linkedin.com/in/faryal-bahawi-28027b205/).
