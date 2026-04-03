# Viva Q&A by Manvendra

---

## Lecture 1 — Introduction to Generative AI & Foundation Models

**Q1: What was the significance of AlexNet in deep learning history?**
A: AlexNet achieved a breakthrough in the ImageNet challenge, demonstrating the power of deep learning for image classification.

**Q2: What is the biological inspiration behind deep learning?**
A: Deep learning is inspired by the structure and functioning of biological neurons in the brain.

**Q3: Name two real-world AI applications mentioned in Lecture 1 besides AlexNet.**
A: AlphaGo (game-playing AI) and Visual Microphone (extracting sound from visual vibrations).

---

## Lecture 2 — Generative Models & GANs

**Q1: What is the key difference between generative and discriminative models?**
A: Discriminative models learn the decision boundary P(y|x), while generative models learn the data distribution P(x) to generate new samples.

**Q2: What are the two components of a GAN and their roles?**
A: Generator creates fake samples to fool the discriminator; Discriminator distinguishes real samples from fake ones.

**Q3: What is the training objective of a GAN?**
A: A minimax game where the generator minimizes and the discriminator maximizes the same loss, converging at Nash Equilibrium.

**Q4: Name three common challenges in GAN training.**
A: Mode collapse, training instability, and vanishing gradients.

**Q5: What is the connection between GAN loss and Jensen-Shannon Divergence (JSD)?**
A: The GAN minimax loss is related to minimizing the Jensen-Shannon Divergence between the real and generated data distributions.

---

## Lecture 3 — Autoencoders & VAEs

**Q1: What is the main problem with a standard autoencoder's latent space?**
A: The latent space is discontinuous — nearby points may not decode to similar outputs, making it unsuitable for generation.

**Q2: How does a VAE differ from a standard autoencoder in encoding?**
A: A VAE encodes input to a distribution (mean μ and std σ) instead of a single point, ensuring a smooth latent space.

**Q3: What is the reparameterization trick in VAEs and why is it needed?**
A: z = μ + σ·ε (where ε ~ N(0,1)); it makes sampling differentiable so gradients can flow through during backprop.

**Q4: What are the two components of the VAE loss (ELBO)?**
A: Reconstruction loss (how well the output matches input) + KL divergence (how close the learned distribution is to a standard normal).

---

## Lecture 4 — Transformers & Attention

**Q1: Why are Transformers preferred over RNNs?**
A: Transformers allow parallelization and handle long-range dependencies better than sequential RNNs.

**Q2: What are Q, K, V in the attention mechanism?**
A: Query (what I'm looking for), Key (what I contain), Value (what I return if matched) — used to compute weighted attention scores.

**Q3: Why do we scale by √dk in scaled dot-product attention?**
A: To stabilize variance of dot products — without scaling, large dk causes extreme softmax values (vanishing gradients).

**Q4: What is the purpose of causal masking in Transformers?**
A: It prevents the model from attending to future tokens during autoregressive generation (e.g., in GPT).

**Q5: What is multi-head attention?**
A: Running multiple attention operations in parallel with different learned projections, then concatenating their outputs.

---

## Lecture 5 — Self-Attention vs Cross-Attention

**Q1: What is the difference between self-attention and cross-attention?**
A: In self-attention, Q/K/V come from the same sequence; in cross-attention, Q comes from one source and K/V from another.

**Q2: Where are MHSA and MHCA placed in the U-Transformer architecture?**
A: MHSA is placed at the bottleneck; MHCA is placed at skip connections.

**Q3: How did U-Transformer perform compared to U-Net?**
A: U-Transformer beat U-Net on all organ segmentation tasks, with especially large gains on the pancreas.

**Q4: What role does positional encoding play in MHCA?**
A: Positional encoding is important in MHCA to preserve spatial information when combining features from different levels.

---

## Lecture 6 — Diffusion Models & Stable Diffusion

**Q1: What happens in the forward diffusion process?**
A: Noise is gradually added to the image over T steps via a Markov chain until it becomes pure Gaussian noise.

**Q2: Write the reparameterized forward diffusion equation.**
A: x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε, allowing direct jump to any timestep t.

**Q3: What does the U-Net predict during reverse diffusion?**
A: The U-Net predicts the noise ε added at each step, which is then subtracted to recover the clean image.

**Q4: What is the training loss for a diffusion model?**
A: Simple MSE between true noise and predicted noise: ||ε − ε_θ(x_t, t)||².

**Q5: What is a Latent Diffusion Model (LDM) and why is it useful?**
A: LDM runs diffusion in VAE-compressed latent space (48× smaller), making it much more computationally efficient.

**Q6: How does Stable Diffusion incorporate text conditioning?**
A: Text embeddings are injected into the U-Net via cross-attention layers during the denoising process.

**Q7: What is Classifier-Free Guidance (CFG)?**
A: A technique that interpolates between conditional and unconditional predictions using a guidance scale w to control text adherence.

---

## Lecture 7 — Embeddings

**Q1: What are the problems with one-hot encoding?**
A: Large weight matrices, huge dataset needs, high computation/memory cost, and no semantic meaning captured.

**Q2: What is the difference between static and contextual embeddings?**
A: Static embeddings (e.g., Word2Vec) give the same vector regardless of context; contextual embeddings (e.g., BERT) change based on surrounding words.

**Q3: How can embeddings be obtained?**
A: Either via dimensionality reduction (e.g., PCA) or by training them as part of the model end-to-end.

---

## Lecture 8 — BERT & ResNet

**Q1: What three types of embeddings are summed to form BERT's input representation?**
A: Token embeddings + Segment embeddings + Position embeddings.

**Q2: What are BERT's two pre-training tasks?**
A: Masked Language Modeling (MLM) with the 80/10/10 masking rule, and Next Sentence Prediction (NSP).

**Q3: What is the residual block formula in ResNet and what problem does it solve?**
A: y = F(x, W) + x — the skip connection solves the vanishing gradient problem in very deep networks.

**Q4: How are BERT and ResNet combined for multimodal sentiment analysis?**
A: BERT processes text and ResNet processes images; their features are combined for sentiment prediction.

---

## Lecture 9 — State Space Models & Mamba

**Q1: What are the two core SSM equations?**
A: h'(t) = Ah(t) + Bx(t) (state update) and y(t) = Ch(t) + Dx(t) (output).

**Q2: What is the dual view of SSMs for training vs inference?**
A: Convolution mode for efficient parallel training; recurrence mode for fast autoregressive inference.

**Q3: What makes Mamba's SSM "selective"?**
A: In Mamba, matrices B, C, and step size Δ become input-dependent functions, allowing the model to selectively focus on relevant information.

**Q4: What are Mamba's key performance advantages over Transformers?**
A: 5× throughput, linear scaling with sequence length, and ability to handle million-token sequences.

---

## Lecture 10 — Contrastive Learning & SimCLR

**Q1: What is contrastive learning?**
A: A self-supervised method that learns representations by pulling positive pairs closer and pushing negative pairs apart.

**Q2: What is the SimCLR pipeline?**
A: Augmentation → encoder f(·) → projection head g(·) → NT-Xent loss.

**Q3: What are SimCLR's three key findings?**
A: Composition of augmentations matters, a non-linear projection head helps, and larger batch sizes improve performance.

**Q4: Why is the projection head discarded after SimCLR training?**
A: The projection head loses information useful for downstream tasks; the encoder representations before it are more general.

---

## Lecture 11 — Transfer Learning & PEFT

**Q1: What are the three stages of the modern LLM pipeline?**
A: Pretraining (language modeling) → Supervised Fine-Tuning (SFT/instruction tuning) → Preference Tuning (RLHF/DPO).

**Q2: Why is transfer learning effective (as shown with medical images)?**
A: A model pretrained on ImageNet reaches 80–90% accuracy on medical images, vs only 60% when trained from scratch.

**Q3: What are adapters and where are they inserted?**
A: Bottleneck modules inserted after self-attention and FFN layers; the original model weights stay frozen.

---

## Lecture 12 — LoRA & QLoRA

**Q1: What is the core idea of LoRA?**
A: Approximate the weight update ΔW as a low-rank decomposition BA (where r << d,k), drastically reducing trainable parameters.

**Q2: How is LoRA initialized and why?**
A: A is random Gaussian, B is zero — so BA = 0 at init, meaning training starts from the original pretrained weights.

**Q3: How does LoRA enable multi-task serving?**
A: Keep one frozen W₀ and swap small (B,A) adapter pairs per task — very low storage and fast switching.

**Q4: What is QLoRA and why is it significant?**
A: QLoRA uses a 4-bit NF4 quantized base model with BF16 LoRA adapters and on-the-fly dequantization — fine-tuning a 7B model in ~5GB memory.

**Q5: What is block-wise quantisation?**
A: Quantising weights in blocks with per-block scaling constants to better handle outlier values and reduce precision loss.

---

## Lecture 13 — RAG (Retrieval-Augmented Generation)

**Q1: What is the RAG pipeline?**
A: Query → embed → vector DB search → dense retrieval → re-rank → augmented prompt → grounded generation → answer with citations.

**Q2: How does dense retrieval differ from keyword search?**
A: Dense retrieval uses text-embedding similarity to capture semantic meaning, not just keyword overlap.

**Q3: What is re-ranking in RAG?**
A: A two-stage process — fast dense retrieval first, then a precise cross-encoder re-scores the top candidates.

**Q4: Name three advanced RAG variants.**
A: Query Rewriting, Multi-query RAG, and Multi-hop RAG.

**Q5: What are key RAG evaluation metrics?**
A: Fluency, Perceived Utility, Citation Recall, Citation Precision, Faithfulness, and Answer Relevance.

---

## Lecture 14 — Flash Attention

**Q1: What is the bottleneck in standard attention — compute or memory?**
A: Memory-bound — the repeated HBM read/writes dominate, not the actual computation.

**Q2: What are the two key techniques in Flash Attention?**
A: Tiling (splitting Q, K, V into blocks processed in SRAM) and recomputation (recomputing S and P in backward pass instead of storing them).

**Q3: What is online softmax and why does Flash Attention need it?**
A: A method using running max m(x) and sum ℓ(x) to compute softmax block-by-block without materializing the full N×N attention matrix.

**Q4: What are Flash Attention's performance gains?**
A: 3× speedup on GPT-2, 15% end-to-end wall-clock improvement, with exact (not approximate) output.

---

## Lecture 15 — Prompt Engineering

**Q1: What do Temperature and Top-p control in LLM generation?**
A: Temperature controls randomness/creativity; Top-p (nucleus sampling) limits token selection to the smallest set whose cumulative probability exceeds p.

**Q2: What are the 7 prompt ingredients?**
A: Persona, Instruction, Context, Format, Audience, Tone, and Data.

**Q3: What is the difference between zero-shot, one-shot, and few-shot prompting?**
A: Zero-shot gives no examples, one-shot gives one example, few-shot gives multiple examples to guide the model's response.

**Q4: What is chain prompting?**
A: Breaking a complex task into a pipeline of sequential prompts (e.g., Name → Slogan → Sales Pitch).

**Q5: What are the pitfalls of LLM-generated prompts?**
A: Feedback loops, intent loss, injection bias, and over-optimisation.
