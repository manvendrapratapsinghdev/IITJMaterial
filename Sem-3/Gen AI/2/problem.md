Assignment: Neural Machine Translation with Cross-Attention 
Topic: English to Spanish Translation via Transformer-based Mechanisms Course: Foundational Models and Generative AI 
1. Objective 
The goal is to implement a translation model that utilizes Cross-Attention to map English source tokens to Spanish target tokens. You will build an architecture where the decoder is "aware" of the entire input sequence through Query-Key-Value interactions, as described in your lecture notes. 
2. Dataset Access 
To maintain efficient training in Colab, use the Anki Spanish-English dataset. 
● Dataset Link: https://huggingface.co/datasets/OscarNav/spa-eng ● Constraint: You must limit your training to the first 50,000 sentence pairs to ensure training completes within a reasonable timeframe on a standard Colab GPU. 
3. Implementation Tasks 
Task 1: Preprocessing & Data Pipeline [1 Mark] 
● Standardization: Convert all text to lowercase and remove punctuation to reduce vocabulary size. 
● Tokenization: You must prepend a [start] token and append an [end] token to every Spanish sentence so the model learns sequence boundaries. 
● Vocabulary Constraint: Use a maximum vocabulary size of 15,000 words and a maximum sequence length of 20 tokens to prevent memory overflow. 
Task 2: The Multi-Head Cross-Attention Layer [2 Marks] 
Implement a custom layer that facilitates information flow between the two languages: 
● Queries (Q): Must be derived from the Spanish (Decoder) sequence. ● Keys (K) & Values (V): Must be derived from the entire English (Encoder) source sequence. 
● Mathematical Logic: Every token in the decoder must attend to all positions/words in the encoder to ensure every generated word is contextually grounded.
Task 3: Encoder-Decoder Architecture [1 Mark] 
Build a simplified Transformer-style model: 
● Encoder: Processes English words into a semantic representation where every token is aware of the others. 
● Decoder: Processes Spanish words and integrates the Cross-Attention layer to "look back" at the English source. 
● Hyperparameter Constraint: Use a hidden dimension d_model of 256 and exactly 4 attention heads. 
Task 4: Training & Greedy Inference [1 Mark] 
● Training Loop: Train the model for exactly 20 epochs using the Adam optimizer. ● Inference: Implement a "greedy search" function. The model should start with the [start] token and iteratively predict the next Spanish word until the [end] token is generated or the maximum length is reached. 
4. Submission Requirements 
Submit a single Colab Notebook containing: 
1. Code: Full implementation of the Attention mechanism, Encoder, and Decoder. 2. Visual Proof: A plot of the Training Loss vs. Validation Loss showing convergence. 3. Translation Samples: Provide 5 examples from the test set showing the "Source English," "Target Spanish," and your "Model Prediction." 
4. Brief Analysis (Max 250 words): Based on the definitions in your lecture, explain why Self-Attention (same input sequence) and Cross-Attention (another source) are both necessary for a model to be effective in translation. 
5. Constraints & Guidelines 
● Due Date: 28/02/2026 till 11:59 P.M. 
● Late Penalty: 1 mark deducted per day after the due date. 
● Originality: DO NOT use LLMs or plagiarize from online repositories; the institute's plagiarism policy is strictly enforced.
