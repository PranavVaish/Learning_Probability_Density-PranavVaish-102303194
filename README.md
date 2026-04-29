# 📘 Assignment  
## Learning Probability Density Functions Using Data Only (GAN-Based Approach)

---

## 👨‍💻 Author

**Pranav Vaish**  
🔗 LinkedIn: https://www.linkedin.com/in/pranavvaish20  

---

## 📌 Overview

This repository presents the implementation of **Assignment–2**, focused on learning an **unknown probability density function (PDF)** of a transformed random variable using a **Generative Adversarial Network (GAN)**.

Unlike traditional approaches, **no analytical or parametric form of the PDF is assumed**.  
Instead, the distribution is learned **purely from data samples**, demonstrating **implicit density modeling using GANs**.

---

## 🎯 Objective

The key goals of this assignment are:

- Apply a **nonlinear transformation** to real-world data  
- Treat the transformed data as drawn from an **unknown distribution**  
- Design and train a **GAN** to model this distribution  
- Approximate the learned PDF using **generated samples**  

---

## 📊 Dataset

- **Dataset:** India Air Quality Data  
- **Source:** Kaggle  
- **Link:** https://www.kaggle.com/datasets/shrutibhargava94/india-air-quality-data  
- **Feature Used:** NO₂ (Nitrogen Dioxide) concentration  
- **File:** `data.csv`  

📌 The dataset is downloaded **programmatically via the Kaggle API** to ensure reproducibility.

---

## 🔁 Step 1: Data Transformation

Each NO₂ value $x$ is transformed using:

$$
z = \mathrm{Tr}(x) = x + a_r \sin(b_r x)
$$

### Parameters

$$
a_r = 0.5 \times (r \bmod 7)
$$

$$
b_r = 0.3 \times ((r \bmod 5) + 1)
$$

Where:
- $r = 102316039$

📌 This transformation introduces:
- Nonlinearity  
- Multimodality  
- Analytical intractability of the PDF  

---

## 🤖 Step 2: Learning the PDF using GAN

Since the PDF is unknown, a **Generative Adversarial Network (GAN)** is used to model it.

### 🔹 Generator (G)
- Input: Noise $ \epsilon \sim \mathcal{N}(0,1) $  
- Output: Fake samples $ z_f = G(\epsilon) $  
- Goal: Generate realistic samples to fool the discriminator  

### 🔹 Discriminator (D)
- Input: Real samples $z$ and fake samples $z_f$  
- Output: Probability of being real  
- Goal: Distinguish real vs generated samples  

---

## 🏗️ GAN Architecture

### Generator

| Layer | Units | Activation |
|------|------|-----------|
| Input | 1 | — |
| Dense | 64 | ReLU |
| Dense | 64 | ReLU |
| Output | 1 | Linear |

### Discriminator

| Layer | Units | Activation |
|------|------|-----------|
| Input | 1 | — |
| Dense | 64 | ReLU |
| Dense | 64 | ReLU |
| Output | 1 | Sigmoid |

---

## 📉 Loss Functions

### Discriminator Loss

$$
\mathcal{L}_D =
- \mathbb{E}[\log D(z)]
- \mathbb{E}[\log(1 - D(G(\epsilon)))]
$$

### Generator Loss

$$
\mathcal{L}_G =
- \mathbb{E}[\log D(G(\epsilon))]
$$

📌 Both networks are trained using **binary cross-entropy loss**.

---

## 📈 Step 3: PDF Approximation

After training:

1. Generate a large number of samples from the generator  
2. Estimate the PDF using **Kernel Density Estimation (KDE)**  

$$
\hat{p}_h(z) =
\frac{1}{Nh}
\sum_{i=1}^{N}
K\left(\frac{z - z_f^{(i)}}{h}\right)
$$

📌 The KDE of generated samples is compared with real transformed data.

---

## 📊 Results

- GAN successfully learns the **underlying distribution**  
- KDE plots show **strong overlap** between real and generated data  
- **Multimodal structure** is effectively captured  

---

## 📝 Observations

### 🔹 Mode Coverage
- Multiple modes are captured  
- Minor **mode dropping** may occur early in training  

### 🔹 Training Stability
- Oscillatory loss behavior (expected in GANs)  
- Stability achieved using a **low learning rate**  

### 🔹 Quality of Generated Samples
- High similarity to real data  
- KDE confirms **accurate PDF approximation**  

---

## 🚀 Key Takeaways

- GANs can model **complex, unknown distributions** without explicit density functions  
- Nonlinear transformations can make PDFs analytically intractable—but still learnable  
- KDE is an effective tool for **validating learned distributions**  
