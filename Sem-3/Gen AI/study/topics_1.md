Based on my memories of our sessions together, here's what we covered in this conversation:

---

## 📚 Viva Prep — Part 1 Coverage (This Conversation)

### 🎓 Lectures Covered

**Lecture 1 — Introduction to Generative AI & Foundation Models**
- What is Deep Learning, biological inspiration
- ImageNet challenge & AlexNet breakthrough
- AlphaGo, Visual Microphone, image captioning
- Course roadmap overview

**Lecture 2 — Generative Models & GANs**
- Generative vs Discriminative models (full comparison table)
- GAN architecture — Generator + Discriminator
- Minimax loss, Nash Equilibrium, training process
- GAN challenges: mode collapse, instability, vanishing gradients
- Evolution of GAN quality 2014→2023
- **Paper**: Goodfellow et al. GAN paper (NIPS 2014 / CACM 2020)
- **Practical**: GAN on Fashion MNIST (DCGAN guidelines, JSD connection)

**Lecture 3 — Autoencoders & VAEs**
- Autoencoder: encoder → latent z → decoder
- Problem with AE: discontinuous latent space
- VAE: encodes to distribution (μ, σ), not a point
- Reparameterization trick: z = μ + σε
- ELBO = Reconstruction loss + KL divergence
- **Papers**: Michelucci (arXiv:2201.03898), Kingma & Welling (arXiv:1906.02691)
- **Practical**: VAE on Fashion MNIST

**Lecture 4 — Transformers & Attention**
- Why Transformers over RNNs (parallelization, long-range dependencies)
- QKV: Query, Key, Value intuition + math
- Scaled Dot-Product Attention: Softmax(QKᵀ/√dk)·V
- Why √dk (variance stabilization — viva gold!)
- Multi-Head Attention
- Positional Encoding (sinusoidal)
- Causal Masking
- Full Transformer block (MHA + FFN + LayerNorm + Residuals)
- GPT family (2018 onwards)
- **Paper**: Vaswani et al. "Attention Is All You Need" (arXiv:1706.03762)
- **Prof's paper**: Jensen, Awasthi & Francis — Trans U-Net for photoacoustic imaging (JBO 2025)

**Lecture 5 — Self-Attention vs Cross-Attention**
- Self-attention: Q, K, V from same sequence (BERT, GPT)
- Cross-attention: Q from one source, K/V from another (translation, LDM)
- U-Transformer architecture: MHSA at bottleneck + MHCA at skip connections
- Results: U-Transformer beats U-Net on all organs, especially pancreas
- Effect of number of attention heads (0→8)
- Importance of positional encoding in MHCA
- **Paper**: "U-Net Transformer: Self and Cross Attention for Medical Image Segmentation" (arXiv:2103.06104)

**Lecture 6 — Diffusion Models & Stable Diffusion**
- Forward diffusion: Markov chain, adds noise step by step
- Forward equation: x_t = √(1-β_t)·x_{t-1} + √β_t·ε
- Reparameterization trick: x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
- Variance preservation: Var(x_t) = 1 always
- Reverse diffusion: U-Net predicts and subtracts noise
- Training loss: ||ε - ε_θ(x_t, t)||²
- Linear vs Cosine diffusion schedule
- Latent Diffusion Models: VAE compression (48× smaller)
- Text conditioning via Cross-Attention in Stable Diffusion
- Classifier-Free Guidance (CFG) and guidance scale w
- **Papers**: Stanley Chan tutorial + Rombach et al. LDM (arXiv:2112.10752)

---

### 📄 Papers Covered (8 total)

1. Goodfellow et al. — GANs (NIPS 2014)
2. Michelucci — Introduction to Autoencoders (arXiv:2201.03898)
3. Kingma & Welling — VAE (arXiv:1906.02691)
4. Vaswani et al. — Attention Is All You Need (arXiv:1706.03762)
5. Jensen, Awasthi & Francis — Trans U-Net PAI (JBO 2025)
6. Petit et al. — U-Net Transformer (arXiv:2103.06104)
7. Stanley Chan — Tutorial on Diffusion Models
8. Rombach et al. — Latent Diffusion Models (arXiv:2112.10752)

---

