# Algorithmic & Mathematical Methodology

**Project:** Spam Email Detection Using Machine Learning  
**Author:** Daniel Richard Xavier  
**Course:** Machine Learning & Natural Language Processing Laboratory  

---

## 1. Feature Representation: Term Frequency-Inverse Document Frequency (TF-IDF)

Text messages cannot be directly ingested by numerical classifiers; they must first be converted into a vector-space model. While simple Bag-of-Words (BoW) records raw term frequencies ($TF$), it suffers from a significant limitation: common grammatical words (e.g., *the*, *is*, *at*, *which*) dominate the feature representations despite conveying minimal semantic information regarding whether a message is spam or legitimate.

To resolve this, we employ **Term Frequency-Inverse Document Frequency (TF-IDF)**.

### Mathematical Formulation

Given a term $t$, document $d$, and corpus $D$ with total document count $N = |D|$:

#### 1. Sublinear Term Frequency ($TF(t, d)$):
To dampen the effect of an unusually high frequency of a single repeated term within an email (a common spam evasion tactic), sublinear scaling is applied:
$$TF_{sub}(t, d) = 1 + \log(TF(t, d)) \quad \text{for } TF(t, d) > 0$$

#### 2. Inverse Document Frequency ($IDF(t, D)$):
The IDF metric measures how much information a word provides across the entire corpus:
$$IDF(t, D) = \log\left(\frac{1 + N}{1 + DF(t)}\right) + 1$$
where $DF(t)$ denotes the document frequency (number of emails containing term $t$). The $+1$ constants represent smooth IDF smoothing, preventing division-by-zero for terms not present in training documents.

#### 3. Composite TF-IDF Weight:
$$TF\text{-}IDF(t, d, D) = TF_{sub}(t, d) \times IDF(t, D)$$

#### 4. L2 Normalization:
Each document vector $\mathbf{v}$ is normalized to unit Euclidean length:
$$\mathbf{v}_{norm} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2} = \frac{\mathbf{v}}{\sqrt{\sum_{i=1}^{M} v_i^2}}$$

### N-Gram Expansion
To capture contextual phrase semantics (such as `"wire transfer"`, `"credit card"`, `"lottery prize"`, `"meeting room"`), we configure the vectorizer to extract both **unigrams** and **bigrams**:
$$\text{ngram\_range} = (1, 2)$$

---

## 2. Model 1: Multinomial Naive Bayes (MNB)

Multinomial Naive Bayes is a probabilistic generative classifier well suited for discrete text classification. It applies Bayes' Theorem under the "naive" conditional independence assumption that the occurrence of each word is conditionally independent of any other word given the class label.

### Bayes' Theorem
Given a document $\mathbf{x} = (x_1, x_2, \dots, x_m)$ where $x_i$ represents the TF-IDF feature of term $i$, the posterior probability of class $c \in \{\text{ham}, \text{spam}\}$ is:
$$P(c \mid \mathbf{x}) = \frac{P(c) \cdot P(\mathbf{x} \mid c)}{P(\mathbf{x})} \propto P(c) \prod_{i=1}^{m} P(w_i \mid c)^{x_i}$$

The predicted class $\hat{c}$ corresponds to the Maximum A Posteriori (MAP) decision:
$$\hat{c} = \arg\max_{c \in \{\text{ham}, \text{spam}\}} \left[ \log P(c) + \sum_{i=1}^{m} x_i \log P(w_i \mid c) \right]$$

### Laplace (Additive) Smoothing
If a specific word $w_i$ does not appear in training emails for class $c$, the raw conditional probability would be zero ($P(w_i \mid c) = 0$), causing the entire product to collapse to zero. To prevent this, Laplace smoothing is introduced:
$$P(w_i \mid c) = \frac{N_{ci} + \alpha}{N_c + \alpha |V|}$$
where:
- $N_{ci}$ is the sum of TF-IDF feature weights for term $i$ in class $c$.
- $N_c$ is the total sum of all feature weights in class $c$.
- $|V|$ is the total vocabulary size.
- $\alpha$ is the smoothing parameter (we configure $\alpha = 0.1$ for fine-grained smoothing).

---

## 3. Model 2: Logistic Regression (LR)

Unlike Naive Bayes (a generative model), Logistic Regression is a **discriminative linear classifier**. It does not model how the words were generated; instead, it directly models the posterior boundary $P(y = 1 \mid \mathbf{x})$ through a logistic sigmoid transformation.

### Mathematical Formulation
Let $\mathbf{x} \in \mathbb{R}^d$ be the TF-IDF feature vector, $\mathbf{w} \in \mathbb{R}^d$ be the learned weight vector, and $b \in \mathbb{R}$ be the bias intercept. The linear logit is:
$$z = \mathbf{w}^T \mathbf{x} + b = \sum_{j=1}^{d} w_j x_j + b$$

The predicted probability of an email being spam ($y = 1$) is given by the Sigmoid function:
$$\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

The probability of an email being legitimate ham ($y = 0$) is:
$$P(y = 0 \mid \mathbf{x}) = 1 - \sigma(z)$$

### Decision Boundary
$$\hat{y} = \begin{cases} 1 \quad (\text{SPAM}) & \text{if } \sigma(z) \ge 0.50 \\ 0 \quad (\text{HAM}) & \text{if } \sigma(z) < 0.50 \end{cases}$$

### Loss Function with L2 Regularization (Ridge)
The parameters $\mathbf{w}$ and $b$ are estimated by minimizing the penalized binary cross-entropy (log-loss):
$$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\sigma(z_i)) + (1 - y_i) \log(1 - \sigma(z_i)) \right] + \frac{1}{2C} \|\mathbf{w}\|_2^2$$
where $C > 0$ is the inverse regularization strength (we configure $C = 1.0$, which balances high generalization without overfitting to rare spam tokens).

---

## 4. Model Selection Justification: Generative vs. Discriminative

| Criterion | Multinomial Naive Bayes | Logistic Regression | Project Observation |
| :--- | :--- | :--- | :--- |
| **Model Type** | Generative ($P(X \mid Y)$) | Discriminative ($P(Y \mid X)$) | Complementary approaches tested |
| **Feature Independence** | Assumes strong conditional independence | Does not assume feature independence | LR models correlated phrases better |
| **Training Complexity** | Single-pass linear frequency counting $\mathcal{O}(N \cdot d)$ | Iterative numerical optimization (LibLinear) | Both train in < 3 seconds on CPU |
| **Probability Calibration** | Tends to produce extreme probabilities (near 0 or 1) | Well-calibrated probabilities via log-odds | LR provides smoother confidence scores |
| **Measured Test F1** | 97.68% | **98.42%** | **Logistic Regression selected as champion** |
