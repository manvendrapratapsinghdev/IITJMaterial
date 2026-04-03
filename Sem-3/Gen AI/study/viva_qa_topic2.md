# Viva Q&A — Topic 2 (Lectures 7–15)

---

## Lecture 7 — Embeddings

### 7.1 One-Hot Encoding & Its Problems

**Q1: What is one-hot encoding and list its five problems?**
A: One-hot encoding represents each token as a sparse binary vector with a single 1. Problems: (1) huge weight matrices, (2) large dataset requirements, (3) high computation cost, (4) high memory usage, (5) no semantic meaning captured.
Two semantically similar words (e.g., "king" and "queen") have the same cosine similarity as two unrelated words — one-hot treats everything as equally different.

**Q2: Why is the lack of semantic meaning the most critical drawback of one-hot encoding?**
A: Without semantic meaning, the model cannot leverage word relationships — it must learn from scratch that "happy" and "joyful" are related.
Dense embeddings solve this by placing similar words close together in vector space, giving the model a semantic head start.

### 7.2 Embeddings as Dense Vectors

**Q3: What are embeddings and what are typical real-world dimensions?**
A: Embeddings are dense, low-dimensional vector representations where semantic similarity maps to geometric proximity in the embedding space.
Typical dimensions are 256, 512, or 1024 — much smaller than vocabulary-sized one-hot vectors but rich enough to encode nuanced meaning.

**Q4: Why are embeddings task-specific?**
A: The "best" embedding depends on the downstream task — embeddings trained for sentiment analysis may not be optimal for machine translation.
Different tasks require different notions of similarity — sentiment cares about positive/negative polarity, while translation cares about syntactic roles.

### 7.3 Obtaining Embeddings

**Q5: What are the two main approaches to obtaining embeddings?**
A: (1) Dimensionality reduction (e.g., PCA) on existing representations, or (2) training embeddings end-to-end as learnable parameters within the model.
End-to-end training is preferred for deep learning as the embeddings co-adapt with the rest of the network for the specific task.

### 7.4 Contextual vs Static Embeddings

**Q6: What is the difference between static (Word2Vec) and contextual (BERT-style) embeddings?**
A: Static embeddings assign the same fixed vector to a word regardless of context — "bank" has one vector whether it means river bank or financial bank.
Contextual embeddings produce different vectors based on surrounding words, so "bank" near "river" and "bank" near "money" get different representations.

**Q7: Why are contextual embeddings superior for most NLP tasks?**
A: Natural language is highly ambiguous — word meaning depends on context. Contextual embeddings capture this, reducing errors on polysemous words.
They also capture syntactic roles: the same word used as a noun vs. verb gets different representations, improving downstream task performance.

---

## Lecture 8 — BERT & ResNet

### 8.1 BERT Architecture & Input

**Q1: What does BERT stand for and what makes it "bidirectional"?**
A: Bidirectional Encoder Representations from Transformers. Unlike GPT (left-to-right), BERT attends to both left and right context simultaneously.
This bidirectionality gives BERT a richer understanding of each token because it sees the full sentence context during encoding.

**Q2: What three embedding types are summed to form BERT's input and why each?**
A: Token embeddings (word identity) + Segment embeddings (which sentence: A or B) + Position embeddings (where in the sequence).
Summing them gives each token a representation that encodes what it is, which segment it belongs to, and where it appears — all in one vector.

### 8.2 BERT Pre-training Tasks

**Q3: What is Masked Language Modeling (MLM) and the 80/10/10 rule?**
A: 15% of tokens are selected for prediction. Of those: 80% replaced with [MASK], 10% replaced with a random token, 10% kept unchanged.
The 10/10 split prevents the model from only learning to predict [MASK] tokens — it must also handle noisy and unchanged inputs, improving robustness.

**Q4: What is Next Sentence Prediction (NSP) and why is it used?**
A: NSP trains BERT to predict whether sentence B actually follows sentence A (50% real pairs, 50% random pairs).
This teaches the model inter-sentence relationships, useful for tasks like question answering and natural language inference that require understanding sentence pairs.

**Q5: How are MLM and NSP combined during BERT pre-training?**
A: The total loss is simply MLM loss + NSP loss, both optimized jointly in the same forward pass.
This multi-task setup forces the model to learn both token-level semantics (MLM) and sentence-level relationships (NSP) simultaneously.

### 8.3 ResNet & Residual Learning

**Q6: What is the residual block formula and what problem does it solve?**
A: y = F(x, W) + x — the skip connection adds the input directly to the output of the convolutional layers.
This solves the vanishing gradient problem in very deep networks: gradients flow directly through the skip connection, even if F's gradients vanish.

**Q7: What is the general formula for deep residual networks?**
A: x_L = x_0 + Σ F(xᵢ, Wᵢ) — the output at any layer L is the initial input plus the sum of all residual functions.
This means the gradient from any layer has a direct path back to the input (gradient highway), enabling training of 100+ layer networks.

**Q8: Why can't regular deep networks (without skip connections) be trained effectively beyond ~20 layers?**
A: Without skip connections, gradients must flow through every layer sequentially — repeated multiplication by small values causes them to vanish.
Paradoxically, deeper plain networks performed worse than shallower ones (degradation problem), which ResNet's skip connections solved.

### 8.4 Multimodal Sentiment Analysis (BERT + ResNet)

**Q9: How are BERT and ResNet combined for multimodal sentiment analysis?**
A: BERT encodes the text input into a semantic representation, ResNet encodes the image into a visual representation, and both are fused for joint sentiment prediction.
This leverages text-level nuance (sarcasm, tone) from BERT and visual cues (facial expressions, scene) from ResNet for more accurate sentiment.

**Q10: Why is multimodal fusion better than unimodal for sentiment analysis?**
A: Text alone can miss visual cues (a smile contradicting negative text = sarcasm), and images alone miss linguistic nuance.
Combining both modalities captures complementary signals, improving accuracy on ambiguous cases where one modality is insufficient.

---

## Lecture 9 — State Space Models & Mamba

### 9.1 SSM Fundamentals

**Q1: What are the two core SSM equations and what does each matrix represent?**
A: h'(t) = Ah(t) + Bx(t) (state update) and y(t) = Ch(t) + Dx(t) (output). A maps state-to-state, B maps input-to-state.
C maps state-to-output, and D is a skip connection from input directly to output. Together they define a continuous-time linear dynamical system.

**Q2: What is discretisation in SSMs and why is the step size Δ important?**
A: Discretisation converts continuous SSM equations to discrete steps using Zero-Order Hold, making them applicable to discrete token sequences.
Step size Δ controls the resolution — small Δ gives fine-grained processing (detail-sensitive), large Δ gives coarser processing (faster, broader context).

### 9.2 Dual View: Recurrence vs Convolution

**Q3: Explain the dual view of SSMs — recurrence mode vs convolution mode.**
A: Recurrence mode processes one token at a time (h_t = Āh_{t-1} + B̄x_t) — sequential but efficient for inference since you only maintain the current state.
Convolution mode unrolls the recurrence into a global convolution kernel — parallelizable on GPUs, making training much faster.

**Q4: Why is the dual view practically important?**
A: You get the best of both worlds: use convolution mode during training (parallel, GPU-friendly) and recurrence mode during inference (constant memory, fast autoregressive generation).
This is impossible with standard Transformers, which have O(N²) compute at both training and inference time.

### 9.3 Mamba's Selective SSM

**Q5: What makes Mamba's SSM "selective" and why does it matter?**
A: In standard SSMs, A, B, C, Δ are fixed (input-independent). In Mamba, B, C, and Δ become functions of the input — they change at every timestep.
This allows the model to selectively remember or forget information based on content — like a learned, input-dependent gating mechanism.

**Q6: How does Mamba's hardware-aware algorithm relate to Flash Attention?**
A: Mamba uses a parallel scan algorithm designed to maximize GPU SRAM usage and minimize HBM transfers — the same memory-hierarchy principle as Flash Attention.
Both avoid materializing large intermediate matrices in slow HBM, instead computing block-wise in fast SRAM for massive speedups.

### 9.4 Mamba Results & Scaling

**Q7: What are Mamba's key performance advantages over Transformers?**
A: 5× higher throughput than similarly-sized Transformers, with linear (not quadratic) scaling in sequence length.
This enables processing million-token sequences that would be computationally infeasible for standard attention-based models.

**Q8: Why does linear scaling make Mamba attractive for long-context tasks?**
A: Transformers scale as O(N²) in both compute and memory — doubling sequence length quadruples cost. Mamba scales as O(N) — doubling length only doubles cost.
For tasks like genomics, long-document understanding, or audio processing where N can be millions, this difference is the key enabler.

---

## Lecture 10 — Contrastive Learning & SimCLR

### 10.1 Contrastive Learning Fundamentals

**Q1: What is contrastive learning and what kind of supervision does it use?**
A: Contrastive learning is a self-supervised method that learns representations by pulling positive pairs (same instance, different augmentations) closer and pushing negative pairs apart.
It requires no labels — the "supervision" comes from data augmentations that define which pairs are positive, making it scalable to massive unlabeled datasets.

**Q2: What defines a positive pair and a negative pair in contrastive learning?**
A: A positive pair is two different augmented views of the same image. A negative pair is augmented views from two different images.
The model learns to encode semantic content (invariant to augmentation) rather than surface-level pixel patterns.

### 10.2 SimCLR Pipeline

**Q3: What is the full SimCLR pipeline step by step?**
A: For each image: create two augmented views → pass both through shared encoder f(·) → pass through projection head g(·) → compute NT-Xent contrastive loss.
The encoder learns general-purpose visual features; the projection head is a temporary space optimized for the contrastive objective.

**Q4: What is the NT-Xent loss formula and what does each part do?**
A: L = -log[exp(sim(zᵢ,zⱼ)/τ) / Σ_k exp(sim(zᵢ,zₖ)/τ)]. Numerator: similarity of the positive pair. Denominator: similarity against all negatives.
τ (temperature) controls sharpness — low τ makes the model focus on hard negatives, high τ makes it treat all negatives more equally.

### 10.3 Three Key Findings

**Q5: Why does composition of augmentations matter more than any single augmentation?**
A: No single augmentation (e.g., just cropping or just color jitter) forces the model to learn robust features — it can cheat by relying on shortcuts.
Composing multiple augmentations (crop + color + blur) removes all easy shortcuts, forcing the encoder to learn genuine semantic representations.

**Q6: Why does larger batch size improve SimCLR performance?**
A: Larger batches provide more negative pairs per positive pair, giving the contrastive loss a harder discrimination task.
This pushes the encoder to learn more fine-grained distinctions — with 8192 batch size, each positive pair is contrasted against ~16K negatives.

### 10.4 Projection Head

**Q7: What is the projection head and why must it be non-linear?**
A: The projection head g(·) is an MLP that maps encoder representations to the space where contrastive loss is computed. Non-linearity (ReLU) is critical.
A linear projection head performs significantly worse — the non-linearity allows the head to absorb augmentation-specific information, keeping the encoder's features more general.

**Q8: Why is the projection head discarded after training?**
A: The projection head learns to discard information not useful for the contrastive task (e.g., color info when color jitter is used).
The encoder representations before the projection head retain all information and transfer better to diverse downstream tasks like classification and detection.

---

## Lecture 11 — Transfer Learning & PEFT

### 11.1 Three-Stage LLM Pipeline

**Q1: What are the three stages of the modern LLM pipeline and what does each produce?**
A: (1) Pretraining (next-token prediction on massive text) → base/foundation model. (2) SFT (instruction tuning on curated data) → instruction-following model.
(3) Preference tuning via RLHF or DPO → aligned model that is helpful, harmless, and honest. Each stage refines behavior further.

**Q2: What is the difference between SFT and preference tuning?**
A: SFT teaches the model to follow instructions using (instruction, response) pairs — it learns the format and task structure.
Preference tuning (RLHF/DPO) uses human preference rankings to fine-tune outputs — it learns which responses humans prefer, improving quality and safety.

### 11.2 Transfer Learning

**Q3: What is transfer learning and why is it effective?**
A: Transfer learning reuses a model pretrained on a large dataset (e.g., ImageNet) and fine-tunes it on a smaller target dataset.
It works because lower layers learn general features (edges, textures) transferable across tasks — only the higher task-specific layers need retraining.

**Q4: What was the performance comparison between training from scratch vs transfer learning for medical images?**
A: Training from scratch on medical images achieved ~60% accuracy. Fine-tuning a model pretrained on ImageNet achieved 80–90% accuracy.
This 20–30% jump demonstrates that even though ImageNet (natural images) differs from medical data, the learned low-level features transfer effectively.

### 11.3 Full Fine-Tuning Problems

**Q5: What are the three main problems with full fine-tuning of large models?**
A: (1) Extremely high compute cost — updating all billions of parameters requires massive GPU resources. (2) Slow training due to full gradient computation.
(3) High storage — each fine-tuned version stores a full copy of all parameters, so N tasks require N × model_size storage.

### 11.4 Adapters (Houlsby et al.)

**Q6: What are adapters and how are they inserted into a Transformer?**
A: Adapters are small bottleneck modules (down-project → nonlinearity → up-project) inserted after self-attention and FFN layers in each Transformer block.
The original pretrained weights are frozen; only the small adapter parameters are trained — typically adding <5% extra parameters.

**Q7: Why is the bottleneck design important in adapters?**
A: The down-projection compresses features to a small dimension, the up-projection restores the original dimension — this keeps the adapter parameter count tiny.
This bottleneck forces the adapter to learn a compact, task-specific transformation rather than memorizing, while keeping the base model intact.

---

## Lecture 12 — LoRA & QLoRA

### 12.1 LoRA Fundamentals

**Q1: What is the core idea behind LoRA?**
A: Instead of updating the full weight matrix W, LoRA learns a low-rank decomposition ΔW = BA where B ∈ ℝ^(d×r) and A ∈ ℝ^(r×k), with rank r << min(d,k).
This drastically reduces trainable parameters — the original W stays frozen, and only the small B and A matrices are learned.

**Q2: What is the LoRA forward pass equation?**
A: h = (W₀ + BA)x — the output is the pretrained transformation W₀x plus the learned low-rank update BAx.
At inference, BA can be merged into W₀, so there's zero additional latency compared to the original model.

**Q3: Why is LoRA initialized with A = random Gaussian and B = zero?**
A: Setting B = 0 means BA = 0 at initialization, so training starts exactly from the pretrained model's behavior.
This ensures the pretrained knowledge is preserved at the start — the adaptation is learned gradually from this stable starting point.

### 12.2 LoRA Efficiency & Multi-Task

**Q4: How efficient is LoRA — give the GPT-3 example.**
A: GPT-3 has 175B parameters. LoRA trains only ~0.01% of them = 0.0175B trainable parameters.
This reduces memory, compute, and storage by orders of magnitude while achieving performance comparable to full fine-tuning.

**Q5: How does LoRA enable efficient multi-task serving?**
A: Keep one frozen W₀ in memory and swap tiny (B,A) adapter pairs per task — each task's adapter is just a few MB.
Switching tasks only requires loading a small adapter, not a full model copy — enabling serving hundreds of tasks from a single base model.

### 12.3 Quantisation Basics

**Q6: What is quantisation and what is the absmax formula?**
A: Quantisation reduces weight precision from FP32 to lower bit-widths (Int8, 4-bit), reducing memory and compute. Absmax: x_quant = round(x / max|x| × (2^(b-1) - 1)).
Trade-off: lower bits save memory but introduce precision loss — the model's accuracy can degrade if quantisation is too aggressive.

**Q7: What is block-wise quantisation and why is it needed?**
A: Instead of one scaling constant for the entire tensor, block-wise quantisation uses separate scaling constants for each block of weights.
This handles outlier values better — a single outlier won't stretch the entire quantisation range, reducing precision loss for the majority of weights.

### 12.4 QLoRA

**Q8: What is QLoRA and how does it combine quantisation with LoRA?**
A: QLoRA stores the base model in 4-bit NF4 precision, trains LoRA adapters in BF16, and uses on-the-fly dequantisation during forward passes.
This enables fine-tuning a 7B parameter LLaMA model in just ~5GB of memory — making LLM fine-tuning accessible on consumer GPUs.

**Q9: What is NF4 (Normal Float 4-bit) and why is it used in QLoRA?**
A: NF4 is a 4-bit data type optimized for normally distributed weights — its quantisation levels are spaced according to the normal distribution.
Since pretrained neural network weights are approximately normally distributed, NF4 minimizes quantisation error compared to uniform 4-bit formats.

**Q10: What is on-the-fly dequantisation in QLoRA?**
A: During forward/backward passes, 4-bit NF4 weights are dequantised to BF16 just before computation, then discarded immediately after.
This keeps storage at 4-bit (small memory footprint) while computation happens at BF16 precision (maintaining training quality).

---

## Lecture 13 — RAG (Retrieval-Augmented Generation)

### 13.1 RAG Pipeline

**Q1: What is the full RAG pipeline from query to answer?**
A: Query → embed query → vector DB search → dense retrieval of relevant chunks → re-rank → augmented prompt (query + retrieved context) → LLM generation → answer with citations.
RAG grounds the LLM's output in actual retrieved documents, reducing hallucinations and enabling access to knowledge not in the model's training data.

**Q2: Why was RAG introduced — what problem does it solve?**
A: LLMs hallucinate (generate plausible but incorrect facts) and have static knowledge (frozen at training cutoff date).
RAG solves both by retrieving up-to-date, domain-specific documents at inference time and grounding the generation in real sources with citations.

### 13.2 Dense Retrieval

**Q3: What is dense retrieval and how does it differ from keyword search?**
A: Dense retrieval encodes both queries and documents into dense embedding vectors, then retrieves by cosine similarity in embedding space.
Unlike keyword/BM25 search (exact term matching), dense retrieval captures semantic similarity — "automobile" retrieves documents about "car" even without the exact word.

**Q4: Why is dense retrieval the preferred retrieval method in modern RAG systems?**
A: Dense retrieval handles synonyms, paraphrases, and conceptual similarity that keyword search misses entirely.
It leverages pretrained language model embeddings that understand meaning, not just surface-level token overlap.

### 13.3 Re-Ranking

**Q5: What is re-ranking and why is it a two-stage process?**
A: Stage 1: Fast dense retrieval returns top-K candidates (e.g., 100). Stage 2: A cross-encoder re-scores each candidate for precise relevance ranking.
Dense retrieval is fast but approximate (bi-encoder). Cross-encoder is slow but accurate (processes query-document pairs jointly) — two stages balance speed and quality.

**Q6: Why can't the cross-encoder be used for initial retrieval?**
A: A cross-encoder processes each query-document pair together — for a million documents, that's a million forward passes per query.
The bi-encoder (dense retrieval) encodes documents offline and only computes one query embedding at inference, enabling sub-second search over millions of documents.

### 13.4 Advanced RAG Variants

**Q7: What are Query Rewriting, Multi-query RAG, and Multi-hop RAG?**
A: Query Rewriting: rephrase the user's query for better retrieval. Multi-query: generate multiple query variants and merge results for broader coverage.
Multi-hop: chain multiple retrieval steps where each step's results inform the next query — needed for complex questions requiring information from multiple sources.

**Q8: What is Query Routing in RAG?**
A: Query Routing directs different types of queries to different retrieval backends or knowledge sources based on the query's nature.
For example, factual queries go to a knowledge base, code queries go to a code repository, and recent events go to a web search — optimizing retrieval quality per query type.

### 13.5 RAG Evaluation

**Q9: What are the six RAG evaluation metrics?**
A: Fluency (readability), Perceived Utility (helpfulness), Citation Recall (are all claims cited?), Citation Precision (are citations relevant?), Faithfulness (does answer match sources?), Answer Relevance (does it answer the question?).
These cover both generation quality (fluency, relevance) and retrieval quality (citation recall/precision, faithfulness).

**Q10: What is the "LLM as a judge" evaluation paradigm?**
A: Using a powerful LLM (e.g., GPT-4) to automatically evaluate the quality of another model's RAG outputs against the above metrics.
This scales evaluation beyond expensive human annotation while correlating well with human judgments on most metrics.

---

## Lecture 14 — Flash Attention

### 14.1 GPU Memory Hierarchy

**Q1: Describe the GPU memory hierarchy relevant to Flash Attention.**
A: SRAM: 19 TB/s bandwidth, ~20 MB capacity (very fast, very small). HBM: 1.5 TB/s bandwidth, ~40 GB capacity (slower, much larger). Main memory: even slower.
The key insight is that standard attention is memory-bound (bottlenecked by HBM read/writes), not compute-bound — so optimizing memory access is more impactful than reducing FLOPs.

**Q2: Why is standard attention memory-bound rather than compute-bound?**
A: Standard attention reads/writes the full N×N attention matrix to HBM 4 times (compute S, softmax, dropout, multiply V) — these transfers dominate runtime.
The actual compute (matrix multiplications) is fast on modern GPUs; the bottleneck is moving data between slow HBM and fast SRAM.

### 14.2 Flash Attention: Tiling & Recomputation

**Q3: What is tiling in Flash Attention?**
A: Tiling splits Q, K, V matrices into small blocks that fit entirely in fast SRAM, computing attention block-by-block without ever materializing the full N×N matrix in HBM.
Each block computes its partial attention scores in SRAM and writes only the final output block back to HBM — dramatically reducing memory traffic.

**Q4: What is the recomputation technique and why does it save memory?**
A: Instead of storing the large N×N attention matrix S and probability matrix P for the backward pass, Flash Attention only saves small statistics (m, ℓ) per block.
During backprop, S and P are recomputed on-the-fly from Q, K, V — trading a small amount of extra compute for massive memory savings (O(N) instead of O(N²)).

### 14.3 Online Softmax

**Q5: What is online softmax and why is it essential for Flash Attention?**
A: Online softmax computes softmax incrementally using running max m(x) and running sum ℓ(x), updating block-by-block as new blocks arrive.
Standard softmax needs the full row to compute the max and sum — online softmax avoids this, enabling block-wise attention without the full N×N matrix.

**Q6: How does block concatenation work — how are partial results merged?**
A: When a new block arrives, the running max m is updated, the previous sum ℓ is rescaled by exp(old_m - new_m), and the new block's contribution is added.
This ensures the final result is mathematically exact — identical to standard attention, not an approximation.

### 14.4 Results & Comparison

**Q7: What speedup does Flash Attention achieve on GPT-2?**
A: 3× speedup on attention computation and 15% end-to-end wall-clock improvement for full GPT-2 training.
Critically, the output is exact — not an approximation — so there's no accuracy trade-off, only pure speed and memory gains.

**Q8: Compare standard attention vs Flash Attention on key metrics.**
A: Standard: O(N²) memory, full N×N matrix in HBM, 4 HBM read/write passes, slow. Flash: O(N) memory, block-wise in SRAM, minimal HBM traffic, faster.
Flash Attention achieves this while being exact (not approximate) — the only trade-off is slightly more FLOPs from recomputation, which is hidden by the memory savings.

**Q9: What improvement does Flash Attention 2 bring over Flash Attention 1?**
A: Flash Attention 2 (arXiv:2307.08691) further optimizes parallelism across attention heads and reduces non-matrix-multiply FLOPs.
It achieves even better GPU utilization by improving work partitioning between thread blocks, getting closer to theoretical maximum throughput.

---

## Lecture 15 — Prompt Engineering

### 15.1 Model Selection

**Q1: What factors should guide model selection for a task?**
A: Consider proprietary vs open source (cost, privacy, control trade-offs) and model size (smaller models like Phi-3 can match larger ones on specific tasks).
Match model capability to task complexity — don't use a 175B model for simple classification that a 3B model handles well.

**Q2: What is the Phi-3 example illustrating about model size?**
A: Phi-3 (a smaller model) demonstrated competitive performance with much larger models on certain benchmarks.
This shows that model architecture and training data quality can compensate for raw parameter count — bigger isn't always better.

### 15.2 Controlling LLM Output

**Q3: What do Temperature and Top-p control in LLM generation?**
A: Temperature controls randomness — low (0.0-0.3) for deterministic/factual output, high (0.7-1.0) for creative/diverse output. Top-p (nucleus sampling) limits tokens to the smallest set whose cumulative probability exceeds p.
With do_sample=False, the model always picks the highest probability token (greedy). With do_sample=True + temperature + top-p, generation becomes stochastic.

**Q4: Give the task-specific settings for email vs creative writing.**
A: Email/translation: low temperature (0.0-0.3), high top-p (~0.9) — predictable, accurate, professional output.
Creative writing/brainstorming: high temperature (0.7-1.0), lower top-p (~0.8) — diverse, surprising, imaginative output with controlled randomness.

### 15.3 Seven Prompt Ingredients

**Q5: What are the 7 prompt ingredients?**
A: Persona (who the LLM should be), Instruction (what to do), Context (background info), Format (output structure), Audience (who it's for), Tone (style/voice), Data (input to process).
Not all 7 are needed for every prompt — but including the relevant ones dramatically improves output quality and consistency.

**Q6: Why is specifying "Persona" in a prompt effective?**
A: Persona activates the model's knowledge associated with that role — "You are a senior oncologist" primes medical expertise, cautious language, and clinical reasoning.
It provides an implicit constraint on vocabulary, depth, and perspective that would otherwise require many explicit instructions.

### 15.4 Prompting Paradigms

**Q7: What is the difference between zero-shot, one-shot, and few-shot prompting?**
A: Zero-shot: no examples, just the instruction — tests the model's general understanding. One-shot: one example to demonstrate the format/style.
Few-shot: multiple examples that establish a pattern — the model learns the task implicitly from the examples, often dramatically improving accuracy on structured tasks.

**Q8: What is chain prompting and give the example from the lecture?**
A: Chain prompting breaks a complex task into a pipeline of sequential prompts, where each step's output feeds into the next step's input.
Lecture example: Generate a product Name → use it to create a Slogan → use both to write a full Sales Pitch. Each step is simpler and more controllable.

### 15.5 LLM-Generated Prompts

**Q9: What are the benefits of using LLMs to generate prompts?**
A: Speed (rapid prompt iteration), consistency (standardized prompt structure), and expansion (LLM can elaborate on terse human instructions into detailed prompts).
This enables non-experts to create effective prompts and allows systematic prompt optimization at scale.

**Q10: What are the four pitfalls of LLM-generated prompts?**
A: (1) Feedback loops — LLM optimizes for what it knows, not what's best. (2) Intent loss — the generated prompt drifts from the user's original goal.
(3) Injection bias — the LLM's own biases get baked into the prompt. (4) Over-optimisation — the prompt becomes too narrow, losing generalization ability.
