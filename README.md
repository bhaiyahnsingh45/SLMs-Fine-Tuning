# 🧠 Fine-Tuning Small Language Models (SLMs) — The Complete Beginner's Guide

> **Target Models:** Gemma 2B/7B, Phi-3 Mini, Llama-3 8B, Mistral 7B, and any model under ~10B parameters  
> **Target Audience:** Beginners to fine-tuning who want a clear mental model before writing any code

---

## 📌 Why This Repo Exists

When you first search "how to fine-tune an SLM," you hit a wall of jargon: LoRA, QLoRA, DPO, RLHF, SFT, PEFT, distillation… It feels like 10 different things but nobody tells you *how they relate to each other*.

**This repo clears that up.** Fine-tuning isn't one thing — it's a combination of choices across two independent dimensions:

| Dimension | The Question It Answers |
|---|---|
| **Training Objective** | *What are you teaching the model?* |
| **Parameter Strategy** | *Which weights do you update — and how?* |

Once you understand these two axes, every technique falls neatly into place.

---

## 🗺️ The Big Picture — A Mental Map

```
Fine-Tuning
│
├── AXIS 1: WHAT are you teaching?  (Training Objective)
│     ├── 1. Continued Pre-Training (CPT)        ← Teach new domain knowledge
│     ├── 2. Supervised Fine-Tuning (SFT)         ← Teach task behavior
│     │       └── Instruction Tuning (IFT)        ← Teach to follow commands
│     └── 3. Preference Alignment                 ← Teach values & tone
│               ├── RLHF
│               ├── DPO
│               └── ORPO
│
└── AXIS 2: HOW are you updating weights?  (Parameter Strategy)
      ├── Full Fine-Tuning (FFT)                  ← Update everything (expensive)
      └── Parameter-Efficient Fine-Tuning (PEFT)  ← Update a tiny fraction (smart)
                ├── LoRA
                ├── QLoRA
                ├── DoRA
                ├── Adapters
                └── Prefix / Prompt Tuning
```

> 💡 **The key insight:** You always combine one choice from Axis 1 with one choice from Axis 2.  
> Example: *"I want to teach my model to follow medical instructions (SFT) but I only have one GPU (QLoRA)"* → you do **SFT with QLoRA**.

---

## 🔑 Decision Tree — "Which Technique Should I Use?"

Start here before reading anything else:

```
START: What is your goal?
│
├── "I want the model to know things it never saw in training"
│     └── → Continued Pre-Training (CPT)
│
├── "I want the model to answer questions / perform a task"
│     └── → Supervised Fine-Tuning (SFT)
│           └── If it also needs to follow commands → add Instruction Tuning
│
├── "The model gives correct answers but in the wrong tone / format"
│     └── → Preference Alignment (DPO or ORPO)
│
├── "The model is good but needs to be safer / more helpful"
│     └── → RLHF (if you have human raters) or DPO (simpler, no raters needed)
│
└── "I want to compress a big model's knowledge into my small model"
      └── → Knowledge Distillation

─────────────────────────────────────────────────────

NOW: What hardware do you have?

├── Multiple A100s / H100s, large dataset, don't mind risk
│     └── → Full Fine-Tuning (FFT)
│
├── 1–2 GPUs (24GB VRAM), serious production use
│     └── → LoRA or DoRA
│
├── Consumer GPU (8–16GB VRAM) or Google Colab
│     └── → QLoRA  ← this is what most people use
│
└── No training at all, just want to steer the model
      └── → Prompt Tuning / Prefix Tuning
```

---

## 📚 Axis 1: Training Objectives (What Are You Teaching?)

### 1. Continued Pre-Training (CPT)

**What it is:** You feed the model raw, unstructured text — like books, manuals, research papers — so it learns new domain knowledge from scratch.

**When to use it:**
- Your domain uses language the base model has never seen (medical jargon, legal clauses, internal code, regional dialects)
- You need the model to "know" something deeply, not just follow instructions about it

**What you need:** Large unlabelled text corpus (tens of thousands of documents)

**Real example:**
```
Goal: Fine-tune Gemma 2B to know your company's internal policy documents.
Data: 50,000 raw policy text files (no Q&A pairs needed).
Method: CPT — just feed raw text and let it predict the next token.
Result: The model now "knows" your company's policies.
```

**Code sketch:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# Load raw text dataset — NO labels needed
dataset = load_dataset("text", data_files={"train": "company_policies/*.txt"})

# Standard causal language modeling (next-token prediction)
trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="./cpt-output", num_train_epochs=3),
    train_dataset=tokenized_dataset,
)
trainer.train()
```

---

### 2. Supervised Fine-Tuning (SFT)

**What it is:** You train on labeled Prompt → Response pairs. The model learns to stop "autocompleting text" and start "answering questions."

**When to use it:** This is **the most common starting point**. Use SFT when you have:
- A specific task (classification, summarization, extraction, Q&A)
- At least 500–1,000 high-quality examples
- A clear input-output format

**Real example:**
```
Goal: Fine-tune Phi-3 Mini to extract drug names from clinical notes.
Input:  "Patient was prescribed Metformin 500mg twice daily."
Output: {"drugs": ["Metformin"], "dosage": ["500mg"], "frequency": ["twice daily"]}
Method: SFT on 2,000 annotated clinical notes.
```

**Dataset format (JSONL):**
```json
{"prompt": "Extract drug info: Patient was prescribed Metformin 500mg twice daily.",
 "response": "{\"drugs\": [\"Metformin\"], \"dosage\": [\"500mg\"]}"}
{"prompt": "Extract drug info: Take Atorvastatin 10mg at bedtime.",
 "response": "{\"drugs\": [\"Atorvastatin\"], \"dosage\": [\"10mg\"]}"}
```

**Code sketch:**
```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",   # field containing "prompt + response"
    max_seq_length=512,
    tokenizer=tokenizer,
)
trainer.train()
```

---

### 2a. Instruction Tuning (IFT) — a sub-type of SFT

**What it is:** A specialized form of SFT where the dataset is designed to teach the model to follow natural language commands ("Summarize this", "Translate to French", "Act as a customer support agent").

**When to use it:** When you want to transform a **base model** (which just autocompletes) into an **instruct model** (which understands and obeys commands).

**Real example:**
```
Before IFT: "The patient has fever..." → model continues: "and cough. The doctor..."
After IFT:  "Summarize this: The patient has fever..." → model: "Patient presents with fever."
```

**Data format (Alpaca-style):**
```json
{
  "instruction": "Summarize the following medical note in one sentence.",
  "input": "The patient, a 45-year-old male, presented with high fever...",
  "output": "A 45-year-old male presented with fever requiring evaluation."
}
```

---

### 3. Preference Alignment

**What it is:** You don't just teach the model *what* to say — you teach it *how to behave*: be more helpful, less toxic, match a specific tone, refuse certain requests.

**When to use it:**
- After SFT, when the model gives correct answers but in the wrong style
- For safety tuning (making the model refuse harmful requests)
- For format alignment (always respond in JSON, always be concise, etc.)

---

#### 3a. RLHF (Reinforcement Learning from Human Feedback)

**How it works:** Humans rate multiple model outputs → a Reward Model is trained on those ratings → the SLM is then optimized to maximize the reward score.

**When to use:** When you have human raters available and need the highest quality alignment. Complex to set up.

**Real example:** OpenAI used RLHF to make ChatGPT helpful and safe. For SLMs it's rarely used directly due to complexity.

---

#### 3b. DPO (Direct Preference Optimization) ⭐ Recommended

**How it works:** You create a dataset of *(prompt, chosen_response, rejected_response)* pairs and train directly on them — no reward model needed. Much simpler than RLHF.

**When to use:** The go-to alignment technique for SLMs. Replaces RLHF in most practical cases.

**Real example:**
```
Goal: Make a customer service bot less robotic.

Prompt:      "My order hasn't arrived."
Chosen:      "I'm sorry to hear that! Let me look into this right away. 
               Could you share your order number?"
Rejected:    "Order not found. Provide order ID."

The model learns to prefer the warmer response.
```

**Dataset format:**
```json
{
  "prompt": "My order hasn't arrived.",
  "chosen": "I'm sorry to hear that! Let me look into this right away...",
  "rejected": "Order not found. Provide order ID."
}
```

**Code sketch:**
```python
from trl import DPOTrainer

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,       # original SFT model (frozen reference)
    beta=0.1,                  # controls deviation from reference
    train_dataset=dataset,
    tokenizer=tokenizer,
)
dpo_trainer.train()
```

---

#### 3c. ORPO (Odds Ratio Preference Optimization) ⭐ Best for SLMs

**How it works:** Combines SFT and preference alignment into a **single training step**. You don't need a separate reference model, which saves memory — critical for SLMs.

**When to use:** When you want to do SFT + alignment in one shot on limited hardware.

**Real example:** Training Gemma 270M to answer banking queries correctly AND politely — in one pass.

```python
from trl import ORPOTrainer, ORPOConfig

orpo_config = ORPOConfig(
    learning_rate=8e-6,
    lambda_=0.1,               # weight for preference loss
    max_length=1024,
    output_dir="./orpo-output"
)
trainer = ORPOTrainer(
    model=model,
    args=orpo_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
```

---

## ⚙️ Axis 2: Parameter Strategies (How Are You Updating Weights?)

### 1. Full Fine-Tuning (FFT)

**What it is:** Every single parameter in the model is updated during training.

**Pros:** Maximum performance ceiling — model can fully adapt.  
**Cons:** Requires massive GPU memory. High risk of *catastrophic forgetting* (model forgets everything it knew before). Very slow.

**When to use:** Only when:
- You have multi-GPU clusters (4×A100 or more)
- Your new task is so different that preserving old knowledge doesn't matter
- You have tens of thousands of training examples

**Real example:**
```
Updating all 7B parameters of Llama-3 on a domain-specific legal dataset.
Requires: ~80GB VRAM minimum. Risk: Model may forget general reasoning.
```

> ⚠️ **For SLMs: Avoid FFT unless you have a clear reason.** The smaller the model, the more severe catastrophic forgetting becomes.

---

### 2. Parameter-Efficient Fine-Tuning (PEFT) — The SLM Standard

**What it is:** The base model's weights are **frozen**. Only a tiny set of new or injected parameters are trained. This preserves the model's existing knowledge while teaching it something new.

**Why it matters for SLMs:**
- Reduces trainable parameters by **99%+**
- Runs on consumer GPUs
- Drastically reduces risk of catastrophic forgetting

---

#### 2a. LoRA (Low-Rank Adaptation) ⭐ Most Popular

**How it works:** Instead of updating a weight matrix `W` directly, LoRA injects two small matrices `A` and `B` (where `rank << original size`) alongside it. Only `A` and `B` are trained.

```
Original: W (e.g., 4096 × 4096 = 16M params)  → FROZEN
LoRA adds: A (4096 × r) + B (r × 4096)         → TRAINED
Where r = 8 or 16 (tiny!)
```

**When to use:** Default choice for most fine-tuning tasks with 1–2 GPUs.

**Real example:**
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                          # rank — higher = more capacity, more memory
    lora_alpha=32,                 # scaling factor
    target_modules=["q_proj", "v_proj"],  # which layers to inject LoRA into
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 3,407,872 || all params: 3,812,864,000 || trainable%: 0.089%
```

---

#### 2b. QLoRA (Quantized LoRA) ⭐ Best for Limited Hardware

**How it works:** QLoRA = LoRA + 4-bit quantization of the base model. The frozen base model is loaded in 4-bit precision (instead of 16-bit), cutting VRAM requirements by ~75%. LoRA adapters are still trained in full precision.

**When to use:** You have a consumer GPU (8–16GB VRAM), or you're using Google Colab free tier.

**Real example:**
```
Fine-tuning Gemma 7B with QLoRA:
- Without QLoRA: needs ~28GB VRAM
- With QLoRA: needs ~8GB VRAM ✅ (fits on a T4 GPU!)
```

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                         # 4-bit quantization
    bnb_4bit_quant_type="nf4",                 # Normal Float 4 — best for LLMs
    bnb_4bit_compute_dtype=torch.float16,      # Computation still in float16
    bnb_4bit_use_double_quant=True,            # Quantize quantization constants too
)

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b",
    quantization_config=bnb_config,
    device_map="auto"
)

# Then apply LoRA on top of quantized model — this is QLoRA
model = get_peft_model(model, lora_config)
```

---

#### 2c. DoRA (Weight-Decomposed Low-Rank Adaptation)

**How it works:** Extends LoRA by decomposing weight updates into *magnitude* and *direction* components separately. The direction is updated with LoRA; magnitude gets a separate scalar. This gives more stable gradients.

**When to use:** When LoRA training is unstable (loss spikes, poor convergence). Acts as an upgrade to LoRA.

```python
lora_config = LoraConfig(
    r=16,
    use_dora=True,   # Just add this flag to your existing LoRA config!
    ...
)
```

---

#### 2d. Adapters

**How it works:** Small bottleneck MLP layers ("adapters") are inserted between existing transformer blocks. Only these tiny new layers are trained; everything else is frozen.

**When to use:** When you need to maintain multiple task-specific variants of the same model cheaply (each task gets its own adapter, all share the same base model).

**Real example:**
```
One Gemma 2B base model.
Adapter_A → trained for summarization
Adapter_B → trained for NER extraction  
Adapter_C → trained for sentiment analysis

Swap adapters at inference time — no model reloading needed.
```

---

#### 2e. Prefix Tuning & Prompt Tuning

**How it works:** The model weights stay **completely frozen**. Instead, a small set of trainable vectors ("soft prompts") are prepended to every input. The model learns to "condition" on these vectors.

**When to use:**
- You cannot modify model weights at all (compliance, deployment constraints)
- You want the smallest possible trainable footprint
- Task is relatively simple

**Limitation:** Generally underperforms LoRA for complex tasks.

```python
from peft import PromptTuningConfig, TaskType

prompt_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=20,   # 20 soft prompt tokens prepended to every input
)
model = get_peft_model(model, prompt_config)
# Only 20 token embeddings are trained — everything else frozen
```

---

## 🔬 Advanced & Hybrid Techniques

### Knowledge Distillation

**What it is:** A large "Teacher" model (e.g., GPT-4, Claude) generates high-quality outputs which are used as training data for the smaller "Student" SLM. The SLM learns to *mimic* the teacher.

**When to use:**
- You don't have enough human-labeled data
- You want an SLM to approximate LLM quality on a specific task
- Dataset generation is cheaper than human annotation

**Real example:**
```
Step 1: Send 10,000 medical questions to GPT-4 → get 10,000 expert answers
Step 2: Fine-tune Gemma 2B on those GPT-4 answers (SFT with QLoRA)
Step 3: Gemma 2B now answers medical questions nearly as well as GPT-4
        at 1/100th the inference cost.
```

```python
# Step 1: Generate teacher responses
import openai

def generate_teacher_data(questions):
    results = []
    for q in questions:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": q}]
        )
        results.append({"prompt": q, "response": response.choices[0].message.content})
    return results

# Step 2: Use this dataset for SFT on your SLM (standard SFT pipeline)
```

---

### Model Merging (Training-Free)

**What it is:** You combine the weights of two separately fine-tuned models into one using mathematical merging (e.g., SLERP, TIES, DARE). No additional training required.

**When to use:**
- You have a coding-expert model AND a reasoning-expert model, and want both capabilities
- You want to quickly experiment without training
- "Ensemble on a budget"

**Real example:**
```
Model A: Gemma-2B fine-tuned on medical Q&A
Model B: Gemma-2B fine-tuned on structured JSON output
Merged:  A single model that answers medical questions in structured JSON format
```

```python
# Using mergekit (pip install mergekit)
# config.yaml:
# merge_method: slerp
# base_model: google/gemma-2b
# models:
#   - model: ./gemma-medical
#   - model: ./gemma-json-output
# dtype: float16
```

---

## 📊 Quick Reference Table

| Technique | Category | Trainable Params | Min VRAM | Best For | Watch Out For |
|---|---|---|---|---|---|
| **CPT** | Objective | All (FFT) or PEFT | 40GB+ (FFT) / 8GB (PEFT) | New domain knowledge | Needs lots of raw text |
| **SFT** | Objective | All or PEFT | Depends on method | Task adaptation | Needs labeled data |
| **IFT** | Objective | All or PEFT | Depends on method | Command-following | Instruction dataset quality |
| **DPO** | Objective | ~1% (with LoRA) | 16GB+ | Tone/style alignment | Needs chosen/rejected pairs |
| **ORPO** | Objective | ~1% (with LoRA) | 8GB+ | SFT + alignment in one shot | Newer, less community support |
| **RLHF** | Objective | All | 40GB+ | Highest quality alignment | Very complex, needs human raters |
| **Full FT** | Parameter | 100% | 40GB+ | Max performance | Catastrophic forgetting |
| **LoRA** | Parameter | ~0.1–1% | 16GB | General purpose PEFT | Need to tune rank `r` |
| **QLoRA** | Parameter | ~0.1–1% | 6–8GB | Consumer GPU training | Slight quality drop vs LoRA |
| **DoRA** | Parameter | ~0.1–1% | 16GB | Unstable LoRA training | Slightly slower than LoRA |
| **Adapters** | Parameter | ~1–5% | 8GB | Multi-task, modular serving | Slight inference overhead |
| **Prompt Tuning** | Parameter | <0.01% | 6GB | Frozen model required | Underperforms for complex tasks |
| **Distillation** | Hybrid | PEFT or FFT | 8GB+ | Data-scarce, quality boost | Needs teacher model API |
| **Model Merging** | Hybrid | None (no training) | 8GB (inference) | Combining capabilities | Hit-or-miss results |

---

## 🛠️ Recommended Stack

| Tool | Purpose |
|---|---|
| [🤗 `transformers`](https://github.com/huggingface/transformers) | Load any model |
| [🤗 `peft`](https://github.com/huggingface/peft) | LoRA, QLoRA, Adapters, Prompt Tuning |
| [🤗 `trl`](https://github.com/huggingface/trl) | SFT, DPO, ORPO, RLHF trainers |
| [`bitsandbytes`](https://github.com/TimDettmers/bitsandbytes) | 4-bit / 8-bit quantization for QLoRA |
| [Unsloth](https://github.com/unslothai/unsloth) | 2–5× faster training, 80% less VRAM |
| [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) | No-code unified fine-tuning UI |
| [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) | Config-file based fine-tuning pipeline |
| [Weights & Biases](https://wandb.ai) | Experiment tracking |

---

## 🏁 The Typical SLM Fine-Tuning Workflow

For most real-world use cases, you'll follow this sequence:

```
Step 1: DEFINE YOUR TASK
        └── What exactly should the model do? (Be laser-specific)
        └── What does success look like? (Define your eval metric)

Step 2: COLLECT & CLEAN DATA
        └── Minimum 500 examples for SFT (1,000+ recommended)
        └── If data is scarce → use Knowledge Distillation from GPT-4/Claude

Step 3: CHOOSE YOUR OBJECTIVE (Axis 1)
        └── New knowledge needed? → CPT first, then SFT
        └── Task behavior? → SFT (+ IFT if command-following)
        └── Tone/safety? → DPO or ORPO (after SFT)

Step 4: CHOOSE YOUR PARAMETER STRATEGY (Axis 2)
        └── Consumer GPU? → QLoRA
        └── Professional GPU, stable task? → LoRA
        └── Unstable training? → DoRA
        └── No GPU budget? → Prompt Tuning

Step 5: TRAIN & EVALUATE
        └── Always compare against the base model
        └── Watch for catastrophic forgetting on general benchmarks

Step 6: OPTIMIZE FOR DEPLOYMENT (Optional)
        └── Quantize to 4-bit/8-bit for faster inference
        └── Merge LoRA adapter into base model weights
        └── Distill further if latency is critical
```

---

## 🌍 Real-World Use Case Examples

### Healthcare — Clinical Note Summarization
```
Model: Gemma 2B
Objective: SFT → teach it to summarize doctor's notes
Parameter strategy: QLoRA (fits on a single T4)
Dataset: 3,000 (note, summary) pairs generated via distillation from GPT-4
Result: 90% reduction in summarization cost vs. GPT-4 API
```

### Banking — Fraud Alert Classification
```
Model: Phi-3 Mini 3.8B
Objective: SFT + ORPO (classify transaction + explain in polite tone)
Parameter strategy: LoRA (r=16, on attention layers)
Dataset: 5,000 labeled transactions (fraud/not fraud) + 1,000 preference pairs
Result: 94% accuracy, deployed on-premise (no data leaves the bank)
```

### Education — Personalized Tutoring Bot
```
Model: Llama-3 8B
Objective: IFT → teach to follow student instructions + DPO for tone
Parameter strategy: QLoRA (student interaction needs fast iteration)
Dataset: Distilled from GPT-4 tutoring sessions + teacher preference ratings
Result: Students rated responses 4.2/5 vs. 3.1/5 for base model
```

---

## 📖 Key Concepts Glossary

| Term | Plain-English Definition |
|---|---|
| **Base Model** | The pre-trained model before any fine-tuning (e.g., raw Gemma-2B) |
| **Instruct Model** | A base model after instruction tuning (e.g., Gemma-2B-it) |
| **Catastrophic Forgetting** | When a model forgets general skills after being trained on something narrow |
| **VRAM** | GPU memory — the main bottleneck for fine-tuning |
| **Rank (r)** | LoRA hyperparameter — higher rank = more capacity = more VRAM |
| **Quantization** | Reducing numerical precision (float32 → int4) to save memory |
| **Adapter** | A small trainable module added to a frozen base model |
| **Soft Prompt** | Trainable vectors prepended to inputs (invisible to users) |
| **Reward Model** | In RLHF: a model trained to score outputs on human preference |
| **Chosen / Rejected** | In DPO: paired outputs where one is preferred over the other |

---

## 📂 Repo Structure

```
slm-fine-tuning/
│
├── 01_continued_pretraining/
│   └── cpt_with_qlora.py
│
├── 02_supervised_fine_tuning/
│   ├── sft_basic.py
│   └── instruction_tuning_alpaca.py
│
├── 03_preference_alignment/
│   ├── dpo_trainer.py
│   └── orpo_trainer.py
│
├── 04_peft_methods/
│   ├── lora_config.py
│   ├── qlora_config.py
│   ├── dora_config.py
│   └── prompt_tuning.py
│
├── 05_advanced/
│   ├── knowledge_distillation.py
│   └── model_merging.py
│
├── datasets/
│   └── sample_sft_dataset.jsonl
│
└── README.md  ← you are here
```

---

## 🚦 TL;DR — The 3 Rules of SLM Fine-Tuning

1. **Default to QLoRA** — unless you have good reasons to do otherwise. It works on consumer hardware, preserves base knowledge, and produces production-quality models.

2. **Define your objective first, hardware second** — Know *what* you're teaching before worrying about *how* to update weights. Many beginners jump to techniques without clarity on the task.

3. **Quality > Quantity** — 1,000 carefully curated training examples will outperform 10,000 noisy ones every time. This matters more for SLMs than LLMs.

---

*Contributions welcome. If you find an error or want to add a technique, open a PR.*
