# Viva Q&A — Topic 1 (Lectures 1–6)

---

## Lecture 1 — Introduction to Generative AI & Foundation Models

### 1.1 Deep Learning & Biological Inspiration

**Q1: What is deep learning and what is its biological inspiration?**
A: Deep learning is a subset of machine learning using multi-layered neural networks to learn hierarchical representations from data.
It is inspired by biological neurons — artificial neurons mimic how synapses transmit weighted signals and activate based on thresholds.

**Q2: Why is the hierarchical nature of deep networks important?**
A: Earlier layers learn low-level features (edges, textures) while deeper layers compose them into high-level abstractions (faces, objects).
This mirrors how the visual cortex processes information in stages from simple to complex patterns.

### 1.2 ImageNet & AlexNet Breakthrough

**Q3: What was the ImageNet challenge and why was AlexNet's result a landmark moment?**
A: ImageNet is a large-scale image classification benchmark with millions of images across 1000 categories.
AlexNet (Krizhevsky et al., 2012) dramatically reduced the error rate using deep CNNs with GPU training, proving deep learning's superiority over handcrafted features.

**Q4: What architectural choices made AlexNet successful?**
A: AlexNet used ReLU activations, dropout regularization, data augmentation, and was one of the first models to leverage GPU-based parallel training.
This combination allowed training a much deeper network than previous approaches without overfitting.

### 1.3 Real-World AI Applications

**Q5: What is AlphaGo and why was it significant for AI?**
A: AlphaGo (DeepMind) defeated the world Go champion using deep reinforcement learning combined with Monte Carlo tree search.
Go has ~10^170 possible board positions, making brute-force search impossible — this demonstrated AI's ability to handle extreme combinatorial complexity.

**Q6: What is the Visual Microphone concept?**
A: It extracts audio from video by analyzing tiny vibrations of objects (like a chip bag) caused by sound waves.
This demonstrates how deep learning can recover imperceptible signals from visual data that humans cannot detect.

**Q7: How does image captioning connect vision and language?**
A: Image captioning models use a CNN encoder to extract visual features and a language decoder (RNN/Transformer) to generate text descriptions.
It is a foundational multimodal task bridging computer vision and natural language processing.

### 1.4 Course Roadmap

**Q8: What is the general progression of topics in this GenAI course?**
A: The course moves from foundational generative models (GANs, VAEs) → core architectures (Transformers, Attention) → modern techniques (Diffusion, LLMs, PEFT, RAG).
This builds understanding from basic generation to state-of-the-art controllable and efficient generation.

---

## Lecture 2 — Generative Models & GANs

### 2.1 Generative vs Discriminative Models

**Q1: What is the fundamental difference between generative and discriminative models?**
A: Discriminative models learn P(y|x) — the decision boundary to classify inputs. Generative models learn P(x) — the underlying data distribution to create new samples.
In practice, discriminative models answer "which class?" while generative models answer "what does this class look like?"

**Q2: Can a generative model be used for classification? How?**
A: Yes — using Bayes' rule, P(y|x) = P(x|y)·P(y)/P(x), a generative model that learns class-conditional distributions can classify.
However, discriminative models are usually more accurate for classification since they directly optimize the decision boundary.

### 2.2 GAN Architecture

**Q3: Describe the two-player game in a GAN.**
A: The Generator G takes random noise z and produces fake data G(z); the Discriminator D receives both real and fake data and outputs a probability of being real.
G tries to fool D by making G(z) indistinguishable from real data, while D tries to correctly identify fakes — they improve each other adversarially.

**Q4: What is the GAN minimax objective function?**
A: min_G max_D [E[log D(x)] + E[log(1 - D(G(z)))]]. D maximizes its ability to distinguish real from fake, G minimizes D's success.
At optimal convergence, D outputs 0.5 for all inputs — it cannot tell real from generated — this is the Nash Equilibrium.

**Q5: What is Nash Equilibrium in the context of GANs?**
A: It is the point where neither G nor D can improve unilaterally — G produces perfect samples and D cannot distinguish them.
In practice, reaching exact Nash Equilibrium is very difficult, which is why GAN training is notoriously unstable.

### 2.3 GAN Training Challenges

**Q6: What is mode collapse in GANs?**
A: Mode collapse occurs when the Generator learns to produce only a few types of outputs instead of the full diversity of the real data distribution.
For example, a GAN trained on digits might only generate 3s and 7s, ignoring all other digits.

**Q7: Why do vanishing gradients occur in GAN training?**
A: If D becomes too strong too fast, D(G(z)) → 0, and the gradient of log(1 - D(G(z))) vanishes — G gets no useful learning signal.
This is why alternating training and careful learning rate balancing between G and D are essential.

**Q8: How did GAN-generated image quality evolve from 2014 to 2023?**
A: In 2014, GANs produced blurry, low-resolution images. By 2023, models like StyleGAN produce photorealistic high-resolution faces indistinguishable from real photos.
Key improvements came from progressive growing, style-based architectures, and better training stabilization techniques.

### 2.4 Practical: DCGAN & JSD

**Q9: What are the DCGAN architectural guidelines?**
A: DCGAN uses strided convolutions instead of pooling, batch normalization in both G and D, ReLU in G (except output: Tanh), and LeakyReLU in D.
These guidelines stabilize training and produce better image quality compared to fully connected GANs.

**Q10: What is the connection between GAN loss and Jensen-Shannon Divergence?**
A: The GAN minimax objective, when D is optimal, reduces to 2·JSD(P_real || P_generated) - 2·log2.
So training G effectively minimizes the JSD between the real and generated distributions — when JSD=0, distributions are identical.

---

## Lecture 3 — Autoencoders & VAEs

### 3.1 Autoencoder Basics

**Q1: What is an autoencoder and what is its structure?**
A: An autoencoder compresses input into a low-dimensional latent representation z via an encoder, then reconstructs it via a decoder.
The bottleneck forces the network to learn the most essential features — it's trained to minimize reconstruction error.

**Q2: Why can't standard autoencoders be used for generation?**
A: The latent space of a standard AE is discontinuous and irregular — small interpolations between latent points produce meaningless outputs.
There's no structure guaranteeing that arbitrary points in latent space decode to valid data, unlike VAEs.

### 3.2 Variational Autoencoder (VAE)

**Q3: How does a VAE's encoder differ from a standard autoencoder's encoder?**
A: A VAE encoder outputs a distribution (mean μ and standard deviation σ) rather than a single point vector.
This forces the latent space to be smooth and continuous — nearby points decode to similar outputs, enabling meaningful generation and interpolation.

**Q4: What is the reparameterization trick and why is it necessary?**
A: z = μ + σ · ε where ε ~ N(0,1). It separates the randomness (ε) from the learnable parameters (μ, σ).
Sampling is non-differentiable, but this trick makes z differentiable w.r.t. μ and σ, allowing backpropagation through the sampling step.

**Q5: What is the ELBO loss and what are its two components?**
A: ELBO = Reconstruction Loss + KL Divergence. Reconstruction loss measures how well the decoder recreates the input from z.
KL divergence regularizes the latent space by penalizing the encoded distribution for deviating from a standard normal N(0,1).

**Q6: What happens if KL divergence weight is too high or too low?**
A: Too high → the model ignores the input and outputs blurry averages (posterior collapse), prioritizing a nice latent space over reconstruction.
Too low → the latent space becomes irregular and unstructured (like a standard AE), losing the ability to generate meaningful new samples.

### 3.3 Papers: Michelucci & Kingma-Welling

**Q7: What does the Michelucci paper (arXiv:2201.03898) cover?**
A: It is an introduction to autoencoders covering the fundamentals: encoder-decoder architecture, bottleneck, latent space, and applications.
It serves as a primer for understanding the transition from basic AEs to variational approaches.

**Q8: What is the main contribution of Kingma & Welling (arXiv:1906.02691)?**
A: They introduced the VAE framework — combining variational inference with neural networks for scalable generative modeling.
Key innovations include the reparameterization trick and the ELBO objective, which made training deep latent variable models practical.

---

## Lecture 4 — Transformers & Attention

### 4.1 Why Transformers Over RNNs

**Q1: What are the two fundamental limitations of RNNs that Transformers solve?**
A: RNNs process tokens sequentially (no parallelization) and suffer from vanishing gradients over long sequences (poor long-range dependencies).
Transformers use self-attention to process all tokens simultaneously and directly connect any two positions regardless of distance.

**Q2: How does parallelization in Transformers improve training efficiency?**
A: Since attention computes all token relationships at once (not step-by-step), the entire sequence can be processed in a single forward pass on GPUs.
This dramatically reduces training time compared to RNNs where each token must wait for the previous one to be processed.

### 4.2 Attention Mechanism — Q, K, V

**Q3: Explain the Query, Key, Value intuition with an analogy.**
A: Query = "what am I looking for?", Key = "what do I contain?", Value = "what do I return if matched?" — like a search engine matching queries to indexed keys.
Each token generates Q, K, V vectors via learned linear projections; attention scores are computed as similarity between Q and K, used to weight V.

**Q4: What is the scaled dot-product attention formula and what does each part do?**
A: Attention(Q,K,V) = Softmax(QK^T / √dk) · V. QK^T computes pairwise similarity scores between all token pairs.
Division by √dk prevents dot products from growing too large (which would push softmax into saturation), and softmax converts scores to probabilities that weight V.

**Q5: Why exactly does scaling by √dk stabilize training?**
A: For random vectors of dimension dk, the variance of their dot product scales as dk. So as dk grows, dot products become very large in magnitude.
Large values push softmax into regions with near-zero gradients (saturation), making learning impossible — dividing by √dk keeps variance at 1.

### 4.3 Multi-Head Attention & Positional Encoding

**Q6: What is multi-head attention and why use multiple heads instead of one?**
A: Multi-head attention runs h parallel attention operations with different learned projections, then concatenates and projects the results.
Different heads can attend to different types of relationships (syntactic, semantic, positional), giving the model richer representational power.

**Q7: Why is positional encoding needed and how does the sinusoidal version work?**
A: Transformers have no inherent notion of token order (unlike RNNs), so positional information must be explicitly added.
Sinusoidal encoding uses sin and cos functions of different frequencies for each dimension, allowing the model to learn relative positions and generalize to unseen sequence lengths.

### 4.4 Transformer Block & Causal Masking

**Q8: What are the components of a full Transformer block?**
A: Multi-Head Attention → Add & LayerNorm (residual connection) → Feed-Forward Network → Add & LayerNorm (another residual connection).
Residual connections prevent gradient degradation in deep stacks, and LayerNorm stabilizes training by normalizing activations.

**Q9: What is causal masking and where is it used?**
A: Causal masking sets future positions to -∞ before softmax, ensuring each token can only attend to itself and previous tokens.
It's used in decoder-only models (GPT) for autoregressive generation — the model predicts the next token without peeking ahead.

### 4.5 GPT Family & Prof's Paper

**Q10: What is the GPT family's core architecture approach (from 2018 onwards)?**
A: GPT models use decoder-only Transformers with causal masking, trained autoregressively to predict the next token.
The key insight was that scaling this simple architecture with more data and parameters leads to increasingly capable language models.

**Q11: What is the Prof's Trans U-Net paper about (Jensen, Awasthi & Francis, JBO 2025)?**
A: It applies Transformer-based attention to U-Net for photoacoustic imaging reconstruction, improving image quality.
This demonstrates how attention mechanisms from NLP can be successfully transferred to medical imaging tasks.

---

## Lecture 5 — Self-Attention vs Cross-Attention

### 5.1 Self-Attention

**Q1: What is self-attention and where is it used?**
A: In self-attention, Q, K, V are all derived from the same input sequence — every token attends to every other token in the same sequence.
Used in BERT (bidirectional self-attention over full sentence) and GPT (causal self-attention attending only to past tokens).

**Q2: What kind of relationships does self-attention capture?**
A: Self-attention captures intra-sequence dependencies — how tokens within the same sequence relate to each other.
For example, resolving pronoun references ("The cat sat on the mat because *it* was tired" — *it* attends to *cat*).

### 5.2 Cross-Attention

**Q3: How does cross-attention differ from self-attention in terms of inputs?**
A: In cross-attention, Q comes from one sequence/source and K, V come from a different sequence/source.
This enables information flow between two different modalities or representations — e.g., text queries attending to image features.

**Q4: Give two practical examples of cross-attention from the lecture.**
A: In machine translation, decoder Q attends to encoder K/V from the source language. In Latent Diffusion Models, the denoising U-Net Q attends to text embedding K/V.
Both cases involve one representation "querying" another for relevant information to guide its output.

### 5.3 U-Transformer Architecture

**Q5: What is the U-Transformer and how does it combine MHSA and MHCA?**
A: U-Transformer extends U-Net by adding Multi-Head Self-Attention (MHSA) at the bottleneck and Multi-Head Cross-Attention (MHCA) at skip connections.
MHSA captures global context at the lowest resolution, while MHCA aligns features between encoder and decoder at each skip connection level.

**Q6: Why is placing MHCA at skip connections beneficial?**
A: Standard U-Net skip connections directly concatenate encoder features with decoder features, which can cause semantic misalignment.
MHCA allows the decoder to selectively attend to relevant encoder features, producing better-aligned feature fusion across resolutions.

### 5.4 Results & Analysis

**Q7: How did U-Transformer compare to U-Net across organs, especially pancreas?**
A: U-Transformer outperformed U-Net on all organ segmentation tasks, with the largest improvement on pancreas segmentation.
Pancreas is the hardest organ to segment (small, variable shape), so attention's ability to capture global context gives the biggest advantage there.

**Q8: What was the effect of increasing the number of attention heads from 0 to 8?**
A: Performance improved as more heads were added, showing that multiple attention heads capture complementary information.
This validates that different heads learn different spatial and contextual relationships useful for segmentation.

**Q9: Why is positional encoding important specifically in MHCA?**
A: Without positional encoding in MHCA, the cross-attention has no spatial reference — it doesn't know where features came from in the image.
Adding positional encoding preserves spatial structure, enabling the model to correctly align features across different resolution levels.

---

## Lecture 6 — Diffusion Models & Stable Diffusion

### 6.1 Forward Diffusion Process

**Q1: What is the forward diffusion process mathematically?**
A: It's a Markov chain that adds Gaussian noise step-by-step: x_t = √(1-β_t)·x_{t-1} + √β_t·ε, where β_t is the noise schedule.
Over T steps, the image gradually becomes pure Gaussian noise — the forward process has no learnable parameters.

**Q2: What is the reparameterized form and why is it useful?**
A: x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε, where ᾱ_t = ∏(1-β_s). This lets you jump directly from x_0 to any x_t in one step.
This avoids computing all T intermediate steps during training — you can sample any random timestep t and get the noised version instantly.

**Q3: Why does variance preservation (Var(x_t) = 1) matter?**
A: It ensures the signal doesn't explode or collapse at any step — the noised image always has a controlled magnitude.
This keeps the neural network's input in a stable numerical range throughout all timesteps, making training more stable.

### 6.2 Reverse Diffusion & Training

**Q4: What does the reverse diffusion process do and what model is used?**
A: Reverse diffusion iteratively denoises x_T (pure noise) back to x_0 (clean image) by predicting and subtracting noise at each step.
A U-Net (conditioned on timestep t) is trained to predict the noise ε_θ(x_t, t) that was added in the forward process.

**Q5: What is the diffusion model training loss?**
A: L = ||ε - ε_θ(x_t, t)||² — simple MSE between the actual noise ε added during forward diffusion and the noise predicted by the U-Net.
This is surprisingly simple compared to the full variational bound — Ho et al. showed this simplified loss works better in practice.

**Q6: What is the difference between linear and cosine noise schedules?**
A: Linear schedule adds noise uniformly across timesteps. Cosine schedule adds noise more slowly at the start and end, faster in the middle.
Cosine schedule preserves more image structure in early steps, leading to better sample quality — used in Improved DDPM.

### 6.3 Latent Diffusion Models (LDM)

**Q7: What is a Latent Diffusion Model and how does it reduce computation?**
A: LDM first compresses images into a much smaller latent space using a pretrained VAE encoder, then runs diffusion in that latent space.
The latent representation is ~48× smaller than pixel space, making diffusion 48× cheaper in memory and compute without losing perceptual quality.

**Q8: How does the VAE connect to the diffusion process in LDM?**
A: The VAE encoder compresses the image to latent z, diffusion adds/removes noise in z-space, and the VAE decoder converts the denoised z back to an image.
The VAE is pretrained and frozen — only the diffusion U-Net is trained, operating entirely in the compressed latent space.

### 6.4 Text Conditioning & Stable Diffusion

**Q9: How does text conditioning work in Stable Diffusion via cross-attention?**
A: Text is encoded into embeddings (via CLIP), then injected into the U-Net's denoising process via cross-attention: Q from image features, K/V from text embeddings.
At every denoising step, the U-Net "asks" the text embedding what to generate, steering the noise removal toward the described content.

**Q10: What is Classifier-Free Guidance (CFG) and what does the guidance scale w control?**
A: CFG combines conditional and unconditional predictions: ε = ε_unconditional + w·(ε_conditional - ε_unconditional). During training, text is randomly dropped to learn both modes.
w=1 gives standard conditional generation; w>1 amplifies text influence (sharper, more text-faithful images); too high w causes artifacts and oversaturation.

**Q11: How is the unconditional model obtained for CFG during training?**
A: During training, the text condition is randomly replaced with a null/empty embedding (e.g., 10% of the time).
This teaches the same model to generate both conditionally and unconditionally — no separate classifier needed, hence "classifier-free."

### 6.5 Papers: Chan Tutorial & Rombach et al.

**Q12: What does the Stanley Chan tutorial cover?**
A: It provides a mathematical walkthrough of diffusion models — forward/reverse processes, noise schedules, and the connection to score matching.
It serves as the theoretical foundation for understanding both DDPM and the extensions used in Stable Diffusion.

**Q13: What is the main contribution of Rombach et al. (arXiv:2112.10752)?**
A: They introduced Latent Diffusion Models — moving diffusion to VAE-compressed latent space, making high-resolution image generation practical.
This became the foundation of Stable Diffusion, combining LDM with CLIP text conditioning and cross-attention for text-to-image generation.
