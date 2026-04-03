**Lecture 7 — Embeddings**
- One-hot encoding and its 5 problems (weights, dataset size, computations, memory, no semantic meaning)
- Embeddings as dense vector representations in embedding space
- Task-specific embeddings, real-world dimensions (256, 512, 1024)
- Obtaining embeddings: PCA vs training as part of the model
- Contextual embeddings (static Word2Vec vs BERT-style contextual)
- Reference: Google ML Crash Course on Embeddings, IBM LLM article

**Lecture 8 — BERT & ResNet**
- BERT: Bidirectional Encoder Representations from Transformers
- BERT input = Token + Segment + Position embeddings (summed)
- Pre-training tasks: MLM (80/10/10 masking rule) and NSP
- Combined loss: MLM + NSP
- ResNet: Deep Residual Learning (He et al., 2015)
- Residual block: `y = F(x, W) + x` — skip connections solving vanishing gradients
- General formula: `x_L = x_0 + Σ F(xᵢ, Wᵢ)`
- Application: BERT (text) + ResNet (image) for Multimodal Sentiment Analysis
- Papers: arXiv:1810.04805, arXiv:1603.08029, arXiv:2412.03625

**Lecture 9 — State Space Models & Mamba** *(covered from reference links)*
- SSM equations: `h'(t) = Ah(t) + Bx(t)` and `y(t) = Ch(t) + Dx(t)`
- Matrices A (state-to-state), B (input-to-state), C (state-to-output), D (skip)
- Discretisation using step size Δ (Zero-Order Hold)
- Dual view: recurrence (inference) vs convolution (training)
- Mamba's selective SSM: B, C, Δ become functions of input
- Hardware-aware parallel scan algorithm (same spirit as Flash Attention)
- Results: 5× throughput vs Transformers, linear scaling, million-token sequences
- Paper: Gu & Dao, arXiv:2312.00752

**Lecture 10 — Contrastive Learning & SimCLR**
- Contrastive learning: extracting representations by contrasting positive and negative pairs
- SimCLR pipeline: augmentation → encoder f(·) → projection head g(·) → NT-Xent loss
- Three key findings: augmentation composition, non-linear projection head, larger batch size
- NT-Xent loss formula: `-log [ exp(sim(zᵢ,zⱼ)/τ) / Σ exp(sim(zᵢ,zₖ)/τ) ]`
- Cosine similarity, temperature τ, indicator function
- Full training algorithm pseudocode
- Why projection head is discarded after training
- Paper: Chen et al., arXiv:2002.05709

**Lecture 11 — Transfer Learning & PEFT**
- Three-stage LLM pipeline: Pretraining → SFT → Preference Tuning
- Language modelling → base/foundation model
- Supervised Fine-Tuning (SFT) = instruction tuning
- Preference tuning (RLHF/DPO) for alignment
- Transfer learning: ImageNet → medical images (60% scratch vs 80-90% fine-tuned)
- Full Fine-Tuning problems: cost, slow training, high storage
- Adapters: bottleneck modules inserted after self-attention and FFN (frozen)
- Paper: Houlsby et al., arXiv:1902.00751

**Lecture 12 — LoRA & QLoRA**
- LoRA: weight update `ΔW = BA` where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r << min(d,k)
- Forward pass: `h = (W₀ + BA)x`
- Initialisation: A = random Gaussian, B = zero (so BA=0 at init)
- GPT-3 example: 175B params, LoRA trains only 0.01% = 0.0175B
- Multi-task serving: one frozen W₀, swap (B₁A₁)/(B₂A₂) per task
- Quantisation: FP32 → Int8/4-bit, absmax formula, precision loss
- Block-wise quantisation: per-block constants to handle outliers
- QLoRA: 4-bit NF4 base + BF16 LoRA adapters + on-the-fly dequant
- Memory: 7B LLaMA fine-tuned in ~5GB with QLoRA
- Papers: arXiv:2106.09685, arXiv:2305.14314

**Lecture 13 — RAG (Retrieval-Augmented Generation)**
- RAG pipeline: query → embed → vector DB search → dense retrieval → re-rank → augmented prompt → grounded generation → answer + citations
- Dense Retrieval (DR): text-embedding similarity (semantic, not keyword)
- Re-ranking: two-stage — fast DR then precise cross-encoder re-scoring
- Benefits: reduce hallucinations, increase factuality, application-specific datasets
- Advanced variants: Query Rewriting, Multi-query RAG, Multi-hop RAG, Query Routing
- RAG evaluation metrics: Fluency, Perceived Utility, Citation Recall, Citation Precision, Faithfulness, Answer Relevance
- "LLM as a judge" evaluation paradigm
- Paper: Lewis et al., NeurIPS 2020

**Lecture 14 — Flash Attention**
- GPU memory hierarchy: SRAM (19 TB/s, 20 MB) → HBM (1.5 TB/s, 40 GB) → Main memory
- Standard attention: 4 HBM read/write steps, O(N²) memory, memory-bound not compute-bound
- Flash Attention: tiling (split Q,K,V into blocks) + recomputation
- Online softmax: running `m(x)`, `f(x)`, `ℓ(x)` — merge across blocks without full S matrix
- Block concatenation update rules (the math from pages 4-5)
- Recomputation: saves `(m, ℓ)` stats, recomputes S and P during backprop
- Results: 3× speedup on GPT-2, 15% end-to-end wall-clock improvement, exact output
- Comparison table: Standard (N×N memory, huge traffic, slow) vs Flash (block-wise, less traffic, faster, still exact)
- Papers: arXiv:2205.14135, arXiv:2307.08691

**Lecture 15 — Prompt Engineering**
- Model selection: proprietary vs open source, smaller vs bigger, Phi-3 example
- Controlling output: Temperature (randomness/creativity) and Top-p (nucleus sampling)
- do_sample = False/True, top-p example with "I am driving a ___"
- Task-specific settings table: email/translation/creative writing/brainstorming
- 7 prompt ingredients: Persona, Instruction, Context, Format, Audience, Tone, Data
- Instruction-based prompting: classification, summarisation, code generation
- Prompting paradigms: Zero-shot, One-shot, Few-shot
- Chain prompting: Name → Slogan → Sales Pitch pipeline
- LLM-generated prompts: benefits (speed, consistency, expansion) vs pitfalls (feedback loop, intent loss, injection bias, over-optimisation)

---

### 📄 Papers Covered in Part 2 (12 total)

| # | Paper | arXiv |
|---|---|---|
| 1 | Devlin et al. — BERT | 1810.04805 |
| 2 | He et al. — ResNet | 1512.03385 |
| 3 | Gu & Dao — Mamba | 2312.00752 |
| 4 | Chen et al. — SimCLR | 2002.05709 |
| 5 | Houlsby et al. — Adapters | 1902.00751 |
| 6 | Hu et al. — LoRA | 2106.09685 |
| 7 | Dettmers et al. — QLoRA | 2305.14314 |
| 8 | Lewis et al. — RAG | NeurIPS 2020 |
| 9 | Dao et al. — Flash Attention | 2205.14135 |
| 10 | Dao et al. — Flash Attention 2 | 2307.08691 |
| 11 | Aghajanyan et al. — Intrinsic Dimensionality | 2012.13255 |
| 12 | Multimodal Sentiment (BERT+ResNet) | 2412.03625 |

---