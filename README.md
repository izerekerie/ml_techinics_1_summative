# MamaCare AI: Maternal Health Chatbot for Africa

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/mamacare-ai/blob/main/MaternalHealthChatbot.ipynb)

A domain-specific chatbot fine-tuned to help pregnant women in Africa access accurate maternal health information and dispel dangerous myths and misconceptions.

---

## Project Overview

Maternal mortality remains a critical challenge in sub-Saharan Africa, where lack of access to accurate health information and prevalence of harmful myths contribute to preventable deaths. **MamaCare AI** addresses this by fine-tuning a Large Language Model (LLM) to serve as a knowledgeable, compassionate maternal health assistant.

### Key Features

- **Myth-Busting**: Identifies and corrects common pregnancy myths prevalent in African communities
- **Evidence-Based Responses**: Provides medically accurate information about pregnancy, childbirth, and postpartum care
- **Culturally Aware**: Responses are tailored for the African context (local foods, traditional practices, healthcare access)
- **Out-of-Domain Handling**: Politely redirects non-maternal-health questions
- **Interactive UI**: Easy-to-use Gradio chat interface

---

## Technical Approach

| Component              | Details                                    |
| ---------------------- | ------------------------------------------ |
| **Base Model**         | TinyLlama-1.1B-Chat-v1.0                   |
| **Fine-Tuning Method** | QLoRA (4-bit quantization + LoRA)          |
| **Library**            | Hugging Face `peft`, `trl`, `transformers` |
| **Dataset Size**       | ~3,000+ instruction-response pairs         |
| **Training Hardware**  | Google Colab Free GPU (T4)                 |
| **UI Framework**       | Gradio                                     |

---

## Dataset

The training dataset combines four sources for comprehensive coverage:

| Source                                   | Size        | Description                                              |
| ---------------------------------------- | ----------- | -------------------------------------------------------- |
| `nashrah18/maternalcareeng`              | 105 rows    | Curated maternal care Q&A pairs                          |
| `ruslanmv/ai-medical-chatbot` (filtered) | ~2,500 rows | Medical dialogues filtered for pregnancy/maternal topics |
| Custom myth-busting dataset              | ~60 rows    | Africa-specific pregnancy myths & misconceptions         |
| Custom maternal health Q&A               | ~60 rows    | Trimester info, nutrition, emergency signs               |

### Preprocessing Steps

1. **Text Cleaning**: Whitespace normalization, special character removal
2. **Deduplication**: Removed duplicate questions
3. **Tokenization**: Using TinyLlama's tokenizer with ChatML template formatting
4. **Length Filtering**: Sequences exceeding 512 tokens removed
5. **Stratified Split**: 90% training, 10% validation

---

## Hyperparameter Experiments

Three experiments were conducted to optimize performance:

| Experiment       | Learning Rate | Batch Size | Epochs | Grad Accum | Description             |
| ---------------- | :-----------: | :--------: | :----: | :--------: | ----------------------- |
| Exp 1 (Baseline) |     2e-4      |     4      |   1    |     4      | Higher LR, single epoch |
| Exp 2            |     5e-5      |     2      |   2    |     8      | Lower LR, more epochs   |
| Exp 3 (Best)     |     1e-4      |     4      |   3    |     4      | Balanced configuration  |

All experiments use:

- **Optimizer**: Paged AdamW 32-bit
- **Scheduler**: Cosine learning rate schedule
- **LoRA Config**: r=16, alpha=32, dropout=0.05
- **Quantization**: 4-bit NF4 with double quantization

---

## 📏 Evaluation Metrics

| Metric     | Base Model | Fine-Tuned | Purpose                           |
| ---------- | :--------: | :--------: | --------------------------------- |
| BLEU-4     |     —      |     —      | N-gram overlap with references    |
| ROUGE-1    |     —      |     —      | Unigram recall                    |
| ROUGE-2    |     —      |     —      | Bigram recall                     |
| ROUGE-L    |     —      |     —      | Longest common subsequence        |
| Perplexity |     —      |     —      | Model confidence (lower = better) |

> _Exact values are populated after running the notebook._

---

## Example Conversations

### Myth-Busting

**User**: Is it true that pregnant women should not eat eggs because the baby will be born bald?

**MamaCare AI**: No, this is a myth. Eggs are an excellent source of protein, iron, and choline, which are essential for your baby's brain development. Eating eggs during pregnancy is healthy and recommended. Your baby's hair growth is determined by genetics, not by what you eat.

### Medical Information

**User**: What are the danger signs during pregnancy?

**MamaCare AI**: Go to the health facility immediately if you experience: severe headache with blurred vision, vaginal bleeding, high fever, severe abdominal pain, convulsions, sudden swelling of face or hands, baby stops moving, or water breaking before labor begins.

### Out-of-Domain

**User**: Can you help me with my math homework?

**MamaCare AI**: I am MamaCare AI, a maternal health assistant. I am specifically designed to help with questions about pregnancy, childbirth, and maternal health. I cannot help with math homework, but I would be happy to answer any questions you have about pregnancy or baby care.

---

## How to Run

### Option 1: Google Colab (Recommended)

1. Click the **Open in Colab** badge above
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells sequentially
4. The Gradio interface will launch with a public shareable link

### Option 2: Local Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/mamacare-ai.git
cd mamacare-ai

# Install dependencies
pip install transformers datasets accelerate peft trl bitsandbytes
pip install rouge-score nltk evaluate gradio matplotlib seaborn

# Run the notebook
jupyter notebook MaternalHealthChatbot.ipynb
```

### Requirements

- Python 3.8+
- CUDA-capable GPU (minimum 8GB VRAM recommended)
- ~5GB free disk space

---

## Repository Structure

```
mamacare-ai/
├── MaternalHealthChatbot.ipynb   # Complete pipeline notebook
├── README.md                     # This file
└── training-logs/                # Generated during training
    ├── experiment_comparison.png
    ├── evaluation_metrics.png
    └── base_vs_finetuned.png
```

---

## Disclaimer

MamaCare AI is an educational project and provides **general health information only**. It is **NOT** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for medical decisions. In case of emergency, contact your local healthcare facility immediately.

---

## References

- [TinyLlama](https://github.com/jzhang38/TinyLlama) - Base model
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [nashrah18/maternalcareeng Dataset](https://huggingface.co/datasets/nashrah18/maternalcareeng)
- [ruslanmv/ai-medical-chatbot Dataset](https://huggingface.co/datasets/ruslanmv/ai-medical-chatbot)

---

## License

This project is for educational purposes. Please respect the licenses of the underlying datasets and models.
