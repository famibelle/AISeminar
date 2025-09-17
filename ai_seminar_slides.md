
# <div style="display: flex; justify-content: center; align-items: center; height: 100vh; text-align: center; font-size: 10vw; font-weight: bold; width: 100%;">AI Seminar for Automotive Experts</div>

---

## Agenda
**14:00-14:30** : Introduction à l'IA & concepts fondamentaux
**14:30-14:50** : Natural Language Processing (NLP) & Large Language Models (LLM)
**14:50-15:10** : Computer Vision et Multimodalité
**15:10-15:30** : Data, Documentation Technique et Code Legacy
**15:30-15:50** : IA appliquée à l'industrie automobile
**15:50-16:00** : Résumé

---

# Médhi Famibelle - AI Engineer & Consultant

---

# Introduction to AI & fundamental concepts​

---
# AI VS GENERATIVE AI​
ARTIFICIAL INTELLIGENCE Artificial Intelligence is a field of computer science that aims to create systems capable of imitating or simulating human intelligence.​

MACHINE LEARNING Machine Learning focuses on building systems that learn and improve from experience without being explicitly programmed.​

DEEP LEARNING Deep Learning uses neural networks with many layers to model complex patterns in data.​

GENERATIVE AI Generative AI can create or generate new content, ideas, or data that resemble human creativity.​

---

# ML: Supervised Learning​

Using Labeled Data​

Classification and Regression Tasks​

![Supervised Learning](https://techvidvan.com/tutorials/wp-content/uploads/sites/2/2020/07/Supervised-Learning-in-ML.jpg)

---

# ML: Supervised Learning​
- Predictive maintenance for vehicle components (e.g., brake pads, tires).
- Driver behavior analysis and risk assessment.
- Traffic sign recognition and classification.
- Lane departure warning systems.

---

# ML: Unsupervised Learning​
Discovering hidden structures​

Clustering and dimensionality reduction techniques​

![Unsupervised Learning](https://media.licdn.com/dms/image/v2/D4D12AQHvfxlwDYDETw/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1703001533708?e=2147483647&v=beta&t=_F7OfRMaoFgrOCusbxIyYHNaxGFRChXVzRMAegpggWg)

---
# ML: Unsupervised Learning​
- Clustering driver behavior patterns for personalized insurance plans.
- Grouping traffic patterns to optimize navigation and route planning.
- Segmenting vehicle usage data to design targeted marketing strategies.

---
# ML: Reinforcement Learning​

<div style="display: flex; align-items: center; gap: 20px;">
  <div style="flex: 1;">
Agents learning through trial and error​

Reward systems​
  </div>

  <div style="flex: 1;">

<iframe width="560" height="315" src="https://www.youtube.com/embed/spfpBrBjntg?si=68Z-oEMzvfxk8p6x&autoplay=1" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

  </div>

</div>
---

# ML: Reinforcement Learning​
- Autonomous driving systems learning optimal driving strategies through simulation.
- Adaptive cruise control systems optimizing fuel efficiency and safety.
- Parking assistance systems learning to navigate complex parking scenarios.

---
# Supervised Learning, Unsupervised Learning, Reinforcement Learning

| Mode            | Labeled Training Data | Definition                                                                                     | Use Cases                                                                                     |
|-----------------|-----------------------|------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| Supervised      | YES                   | During the training phase, the desired outcome is known                                       | Image recognition, value prediction, diagnostics, fraud detection                          |
| Unsupervised    | NO                    | During the training phase, the desired outcome is unknown                                     | Customer segmentation, KPI determination, grouping objects that appear to share similarities |
| Reinforcement   | Depends               | The expected result is evaluated on a case-by-case basis                                      | Recommendation engines, AI in gaming                                                        |

---

# Biological Neurons  

<div style="display: flex; align-items: center; gap: 20px;">
  <div style="flex: 1;">

  **Structure:**  
  - Dendrites  
  - Soma  
  - Axon  

  **Functioning of Synapses:**  
  - Chemical and electrical signal transmission  

  </div>

  <div style="flex: 1;">

  ![Neuron Structure](https://www.researchgate.net/profile/Christos-Pliatsikas/publication/376253955/figure/fig1/AS:11431281218483806@1705590629078/Neuron-anatomy-Created-with-BioRendercom.png)  
  *Illustration of a biological neuron*

  </div>

</div>

---

# Artificial Neurons  
<div style="display: flex; align-items: center; gap: 20px;">
  <div style="flex: 1;">

Mathematical model of the artificial neuron  

Activation functions: ReLU, Sigmoid, Tanh  

Similarities and differences with biological neurons?  

</div>

<div style="display: flex; align-items: center; gap: 20px;">
  <div style="flex: 1;">

  ![Artificial Neuron Structure](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c6/Artificial_neuron_structure.svg/1024px-Artificial_neuron_structure.svg.png)  
  *Illustration of an artificial neuron*
  </div>
</div>


---

# Artificial Neural Networks

Artificial neural networks (ANNs) are computational models inspired by the structure and functioning of biological neural networks. 

They consist of interconnected layers of artificial neurons, where each neuron processes inputs, applies an activation function, and passes the output to the next layer. 

**ANNs** are widely used for tasks such as pattern recognition, classification, and regression in various domains.

---

# Parameters and Weights in Neural Networks

In neural networks, **parameters** refer to the adjustable values that the model learns during training. These include:

**Weights:**

- Represent the strength of the connection between neurons.
- Adjusted during training to minimize the error between predicted and actual outputs.

**Biases:**
- Added to the weighted sum of inputs to shift the activation function.
- Helps the model fit the data better by allowing flexibility in decision boundaries.

---

# Parameters and Weights in Neural Networks

<div style="display: flex; align-items: center; gap: 20px;">
  <div style="flex: 1;">

**Why They Matter:**

- Weights and biases are the core components that enable neural networks to learn patterns and make predictions. By iteratively updating these values using optimization algorithms like gradient descent, the network improves its performance on the given task.
  </div>

  <div style="flex: 1;">
**Example:**

- In a simple neural network, if the input is `x`, the weight is `w`, and the bias is `b`, the output of a neuron is calculated as:
$$\text{output} = \text{activation\_function}(w \cdot x + b)$$

  </div>
</div>
---

# Mistral 7B: Number of Parameters

<div style="display: flex; align-items: center; gap: 20px;">

  <div style="flex: 1;">

The Mistral 7B model is a state-of-the-art foundation model with **7 billion parameters**.


**Comparison:**
- **GPT-4:** Estimated 175 billion parameters.
- **LLaMA 2 (13B):** 13 billion parameters.

  </div>

  <div style="flex: 1;">

![Model Parameters Comparison](https://www.geeky-gadgets.com/wp-content/uploads/2023/09/New-Mistral-7B-instruct-model-from-Mistral-AI.webp)

  </div>
</div>
---

# Multi-Layer Perceptron (MLP)
<div style="display: flex; align-items: center; gap: 20px;">

  <div style="flex: 1;">

A Multi-Layer Perceptron (MLP) is a type of artificial neural network composed of an input layer, one or more hidden layers, and an output layer. Each layer consists of interconnected nodes (neurons) where inputs are processed through weighted connections, activation functions, and biases. 

MLPs are widely used for supervised learning tasks such as classification and regression, leveraging their ability to model complex, non-linear relationships in data.

The concept of the MLP was first introduced in 1969 by Marvin Minsky and Seymour Papert in their book *Perceptrons*, which laid the groundwork for neural network research.

  </div>

  <div style="flex: 1;">

![MLP](https://media.licdn.com/dms/image/D5612AQG2n-h9rBE2NA/article-cover_image-shrink_600_2000/0/1701597139460?e=2147483647&v=beta&t=kTHU5V1z66QpFeikBYqQ4Gwgu-o3V8DlwKWOub6Rr2M)

  </div>

</div>
---

# Backpropagation

<div style="display: flex; align-items: center; gap: 20px;">

  <div style="flex: 1;">

Backpropagation, is a supervised learning algorithm used to train artificial neural networks. It adjusts the weights of the network by propagating the error from the output layer back to the input layer. The process involves two main steps:

**Forward Pass:** Compute the output of the network and the error by comparing the predicted output to the actual target.

**Backward Pass:** Calculate the gradient of the error with respect to each weight using the chain rule and update the weights to minimize the error.

This iterative process continues until the network converges to an optimal solution, making it a cornerstone of deep learning.

  </div>

  <div style="flex: 1;">

![Backpropagation](https://i.makeagif.com/media/9-23-2021/ZbBtF9.gif)

  </div>

</div>
---

# Gradient Descent

<div style="display: flex; align-items: center; gap: 20px;">

  <div style="flex: 1;">

Gradient Descent is an optimization algorithm used to minimize a function by iteratively moving in the direction of the steepest descent, as defined by the negative of the gradient. It is widely used in machine learning to optimize model parameters by reducing the error between predicted and actual values.

**Steps:**
- Initialize parameters (e.g., weights) with random values.
- Compute the gradient of the loss function with respect to the parameters.
- Update the parameters by subtracting the gradient scaled by a learning rate.
- Repeat until convergence or a stopping criterion is met.

**Key Concepts:**
- **Learning Rate:** Controls the step size in each iteration.
- **Convergence:** Achieved when the gradient approaches zero or the loss stops decreasing.

  </div>

  <div style="flex: 1;">
![Gradient Descent](https://blog.datumbox.com/wp-content/uploads/2013/10/gradient-descent.png)
  </div>

</div>

---

# Building an AI Model: Key Steps

<div style="display: flex; align-items: center; gap: 20px;">

  <div style="flex: 1;">


**Define the Problem:**
- Clearly identify the objective and the problem you want to solve.
- Example: Predict brake pad wear or classify traffic signs.

**Collect the Data:**
- Gather relevant and high-quality data for training and testing.
- Example: Sensor data, images, or historical logs.

**Choose a Model:**
- Select an appropriate algorithm or architecture based on the problem.
- Example: Use YOLO for real-time object detection or LSTM for time-series predictions.

**Train the Model:**
- Split the data into training, validation, and test sets.
- Train the model using the training data and evaluate its performance on the validation set.
- Fine-tune hyperparameters to optimize accuracy and generalization.

**Evaluate and Deploy:**
- Test the model on unseen data to ensure robustness.
- Deploy the model in the target environment (e.g., embedded systems, cloud).

  </div>

  <div style="flex: 1; align=right;">

```mermaid
graph TD
   A[Define the Problem] --> B[Collect the Data]
   B --> C[Choose a Model]
   C --> D[Train the Model]
   D --> E[Evaluate and Deploy]
   E --> F[Monitor and Improve]
```
  </div>
</div>
---
# Quiz: Neural Networks

---

# What is the primary advantage of using artificial neural networks (ANNs) in AI applications?
  - A) They are inspired by biological neurons.
  - B) They can model complex, non-linear relationships in data.
  - C) They require no training data.
  - D) They are faster than all other machine learning models.

---

# What is the purpose of backpropagation in training neural networks?
  - A) To initialize the weights of the network.
  - B) To propagate the input data forward through the network.
  - C) To adjust the weights by minimizing the error using gradient descent.
  - D) To add more layers to the network.

---

# What does the gradient descent algorithm aim to achieve in neural network training?
  - A) Maximize the loss function.
  - B) Minimize the loss function by iteratively updating the weights.
  - C) Increase the learning rate.
  - D) Reduce the number of neurons in the network.
---

# Which of the following is a key advantage of using reinforcement learning in neural networks?
  - A) It requires labeled training data.
  - B) It learns optimal strategies through trial and error.
  - C) It is only applicable to supervised learning tasks.
  - D) It does not use reward systems.

---
# MNIST Dataset: Handwritten Digit Recognition

<div style="display: flex; align-items: center; gap: 20px;">

  <div style="flex: 1;">

The MNIST (Modified National Institute of Standards and Technology) dataset is a widely used benchmark in machine learning and computer vision. It consists of 70,000 grayscale images of handwritten digits (0–9), each of size 28x28 pixels. The dataset is used to train and evaluate models for digit recognition tasks.

**Significance:**
- MNIST serves as a starting point for testing and comparing machine learning algorithms.
- It helps in understanding how neural networks can classify numbers based on pixel patterns.

**History:**
- MNIST was introduced in 1998 by **Yann LeCun, Corinna Cortes, and Christopher J.C. Burges** as part of their research on neural networks and machine learning.

**Applications:**
- Digit recognition in postal systems.
- Foundational experiments in deep learning.

  </div>
  <div style="flex: 1;">


![MNIST Dataset Example](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

Example of handwritten digits recognition from the MNIST dataset

  </div>
</div>


---

<div style="display: flex; align-items: center; gap: 20px;">

  <div style="flex: 1;">

# Winter is Coming

  </div>

  <div style="flex: 1;">

  <img src="http://www.quickmeme.com/img/33/331a33b15a910ec8d385e50bedeb974d993c0d448678ca7e0f9d7635a1db54c7.jpg" alt="Winter is Coming Meme" style="width: 50%; border: 1px solid #ccc; border-radius: 8px;">

  </div>
</div>

---
# Tokenization

Tokens in natural language processing (NLP) are like syllables in poetry. Just as syllables are the building blocks of rhythm and structure in a poem, tokens are the fundamental units that allow AI models to process and understand text. 

## "Winter is Coming" → 5 tokens
- **Syllables in a poem:** Win / ter / is / com / ing.
- **Tokens in NLP:** ["Win", "##ter", "is", "Com", "##ing"].

---
# Token in AI Models

The token limit defines the maximum number of tokens a model can process in a single input. Higher token limits enable handling longer contexts, making models more effective for tasks like summarization, code analysis, and document generation.

| Model          | Max Size (tokens)  | Approx. Paperback Pages |
|----------------|---------------------|--------------------------|
| GPT-5          | 128,000            | ~512                    |
| Llama 3.1      | 128,000            | ~512                    |
| Mistral Large  | 64,000             | ~256                    |

---
# Embedding

## Transforming Tokens into Numerical Representations

<div style="display: flex; align-items: center; gap: 20px;">

  <div style="flex: 1;">

Embedding transforms tokens into vectors, which serve as the true input points for the LLM.

  </div>

  <div style="flex: 1;">

![Embedding Example](https://causewriter.ai/wp-content/uploads/2023/08/image-2.png)

  </div>
</div>
---

# How Tokenization and Embedding Work Together:
**Tokenization:**
- Splits text into tokens (e.g., words, subwords, or characters).
- Example: "Winter is coming" → ["Win", "##ter", "is", "com", "##ing"].

**Embedding:**
- Maps each token to a high-dimensional vector in a continuous space.
- Example: ["Win", "##ter", "is", "com", "##ing"] → [[0.12, 0.45, ...], [0.34, 0.67, ...], [0.89, 0.23, ...]].

---

# Why Embedding is Important:
- **Semantic Understanding:** Tokens with similar meanings have closer embeddings in the vector space.

```mermaid
graph LR
  A["Input Phrase: 'Winter is coming'"] --> B["Tokenization: ['Win', '##ter', 'is', 'com', '##ing']"]
  B --> C["Embedding: Dense Numerical Vectors"]

  C["Tokenization Output"]
  C --> D["Token: 'Win'"]
  D --> D1["Vector: [0.12, 0.45, 0.78, ...]"]
  C --> E["Token: '##ter'"]
  E --> E1["Vector: [0.34, 0.67, 0.89, ...]"]
  C --> F["Token: 'is'"]
  F --> F1["Vector: [0.56, 0.23, 0.91, ...]"]
  C --> G["Token: 'com'"]
  G --> G1["Vector: [0.78, 0.12, 0.34, ...]"]
  C --> H["Token: '##ing'"]
  H --> H1["Vector: [0.45, 0.89, 0.67, ...]"]
  ```

---

# <div style="display: flex; justify-content: center; align-items: center; height: 100vh; text-align: center; font-size: 10vw; font-weight: bold; width: 100%;">King - Man + Woman = Queen ?</div>

---

# Embedding Example: 
## King - Man + Woman = Queen

Word embeddings capture semantic relationships between words by representing them as vectors in a high-dimensional space. 

A famous example of this is the analogy:

**"King - Man + Woman = Queen"**

## Explanation:
- The vector difference between "King" and "Man" represents the concept of masculinity.
- Adding the vector for "Woman" shifts the representation to the feminine counterpart, resulting in "Queen."

---

# Embedding Example: 
## King - Man + Woman = Queen

<div style="display: flex; align-items: center; gap: 20px;">

  <div style="flex: 1;">

## Mathematical Representation:
If `v(King)`, `v(Man)`, and `v(Woman)` are the embeddings for "King," "Man," and "Woman," then:

$v(King) - v(Man) + v(Woman) ≈ v(Queen)$


## Why This Works:
- Embeddings encode semantic and syntactic relationships.
- Similar concepts are closer in the vector space, enabling analogies like this.

  </div>

  <div style="flex: 1;">

<div style="text-align: center;">
  <img src="https://dfzljdn9uc3pi.cloudfront.net/2022/cs-964/1/fig-6-2x.jpg" alt="image" style="width: 75%;">
</div>

  </div>
</div>

---

# <div style="display: flex; justify-content: center; align-items: center; height: 100vh; text-align: center; font-size: 10vw; font-weight: bold; width: 100%;">Predicting the Next Word: "Paris is the city of ..."</div>

---

# Predicting the Next Word: "Paris is the city of ..."

**Tokenization:**
- The input sentence "Paris is the city of ..." is tokenized into smaller units: `["Paris", "is", "the", "city", "of"]`.

**Contextual Embedding:**
- Each token is converted into a high-dimensional vector using embeddings, capturing its meaning and context.

---

# Predicting the Next Word: "Paris is the city of ..."

**Probability Distribution:**
- The model computes a probability distribution over the vocabulary for the next word. For "Paris is the city of ...", the probabilities might look like:
  - **"love"**: 0.65
  - "light": 0.20    
  - "art": 0.10
  - Other words: 0.05

**Prediction:**
- The word with the highest probability is selected as the next word.


---
## Predicting the Next Word: "Paris is the city of ..."

# Why This Works:

- **Training Data:** The model has seen similar phrases during training, such as "Paris is the city of **love**."
- **Context Understanding:** The embeddings ensure the model considers the entire sentence context.

---

## Predicting the Next Word: "Paris is the city of ..."
Chart
```mermaid
graph LR
  A["Paris is the city of ..."] --> B["Tokenization"]
  B --> C["Contextual Embedding"]
  C --> D["Probability Distribution"]
  D --> D1["love: 0.65"]
  D --> D2["light: 0.20"]
  D --> D3["art: 0.10"]
  D --> D4["Other words: 0.05"]
```
---

# Predicting the Next Word: "Paris is the city of ..."

## Example Output:
- Input: "Paris is the city of ..."
- Output: "love"

This process demonstrates how language models use context and learned patterns to generate coherent and contextually relevant text.

---

<div style="display: flex; justify-content: center; align-items: center; height: 100vh; text-align: center; font-size: 10vw; font-weight: bold; width: 100%;">Quiz: Tokens and Embedding</div>

---

# What is the purpose of tokenization in natural language processing (NLP)?
- A) To convert text into numerical vectors.
- B) To split text into smaller units like words or subwords.
- C) To train a neural network on text data.
- D) To generate embeddings for tokens.

---
# How do embeddings help in understanding the meaning of tokens?
- A) By splitting tokens into smaller parts.
- B) By mapping tokens to high-dimensional vectors that capture semantic relationships.
- C) By converting tokens into binary representations.
- D) By reducing the size of the vocabulary.


---

# Large Language Models

---

# Large Language Models

Large Language Models (LLMs) are artificial intelligence models, pre-trained on vast text corpora, capable of understanding and generating natural language.

## Key Features:
- **Size:** Their size (billions of parameters) enables them to capture the nuances of human language.
- **Versatility:** They can be adapted to a wide variety of domains and applications.
- **Generalization Capability:** LLMs use deep learning techniques to learn universal linguistic structures and relationships.

---

# Large Language Models

## Examples of Models:
- **GPT-5 (2025), GPT-5o (2025), o2 (2025):** Advanced models for text generation and reasoning.
- **Claude 3(Anthropic, 2025):** A model focused on explanatory reasoning and safety.
- **Gemini 2 (Google, 2025):** A cutting-edge multimodal model for processing text, images, and videos.
- **LLaMA  (Meta, 2025):** An open-source model optimized for diverse applications.
- **Mistral Mixtral (2025):** A French model specialized in natural language processing and content generation.
- **Kyutai 2025:** An advanced multilingual model for speech recognition and contextual understanding.
- **Whisper:** A robust speech transcription model supporting numerous dialects.

---

# Large Language Models

## Applications:
- Translation, summarization, creative writing.
- Code generation, semantic analysis.
- Tackling unseen tasks through **zero-shot learning**.

---
# Data Requirements for Training Large Models

## Volume
- Large-scale datasets are essential to train models with billions of parameters.
- Example: GPT-4 was trained on hundreds of terabytes of text data.

---
## Data Requirements for Training Large Models

# Diversity
- Data should cover a wide range of topics, languages, and domains.
- Example: Text, images, audio, and code for multimodal models.

---
## Data Requirements for Training Large Models

# Quality
- High-quality, clean, and well-annotated data ensures better model performance.
- Example: Removing duplicates, correcting errors, and ensuring balanced representation.

---
## Data Requirements for Training Large Models

#Relevance
- Domain-specific data is critical for fine-tuning models for specialized applications.
- Example: Automotive manuals, sensor logs, and traffic data for autonomous driving.

---
## Data Requirements for Training Large Models

# Sources of Data:

- **Public Datasets:** Common Crawl, Wikipedia, ImageNet.
- **Proprietary Data:** Internal documents, customer interactions, telemetry data.
- **Synthetic Data:** Generated data to augment training sets and cover edge cases.

---
## Data Requirements for Training Large Models

# Challenges:

- **Bias and Fairness:** Ensuring data is representative and unbiased.
- **Privacy:** Complying with regulations like GDPR when using sensitive data.
- **Scalability:** Managing and processing massive datasets efficiently.

---
## Data Requirements for Training Large Models

# Example in Automotive:

- **Data Types:** Sensor data, traffic patterns, driver behavior logs.
- **Use Case:** Training models for predictive maintenance, autonomous driving, and voice assistants.

---

# Infrastructure Requirements for Training LLMs

Training a large language model (LLM) demands advanced infrastructure and significant computational resources. Key requirements include:

## High-Performance Hardware
- **GPU/TPU Clusters:** Specialized hardware for parallel processing and efficient training of deep learning models.
- **Massive Compute Power:** Example: GPT-4 was trained using hundreds of petaflops per day.

---

## Infrastructure Requirements for Training LLMs

# Energy Consumption
- **Global Impact:** 20% of the world's energy is projected to be consumed by AI systems (source: DeepLearning.ai).

---

# Energy Consumption for Training Mistral Large 2

The environmental footprint of training Mistral Large 2: as of January 2025, and after 18 months of usage, Large 2 generated the following impacts: 
- **20.4 ktCO₂e:** Total carbon dioxide equivalent emissions.
- **281,000 m³ of water consumed:** Total water usage.
- **660 kg Sb eq:** Standard unit for resource depletion.

source: [Our contribution to a global environmental standard for AI](https://mistral.ai/news/our-contribution-to-a-global-environmental-standard-for-ai)

| Duration | Estimated Energy Consumption | Equivalent in Nuclear Reactors (1.3 GW) |
|---------------------|-----------------------------|-----------------------------------------|
| 18 months | ≈ 1,073.7 GWh (≈ 1.074 TWh)                      | ≈ 0.0628 reactor (≈ 6.3% of a reactor) |


---

<img src="https://cms.mistral.ai/assets/ee83637f-9f22-4e54-b63f-86277bea2a69.jpg" alt="Mistral AI Environmental Impact" width="50%" style="display: block; margin: 0 auto;">

<small style="font-size: smaller;">Environmental impact of training Mistral Large 2, including carbon emissions, water usage, and resource depletion.</small>

---

# Concrete Applications of LLMs

## 📝 Text Generation
- News & financial reports in real time  
- Creative co-writing (ads, games, scripts)  
- Dynamic technical documentation  

---

## Concrete Applications of LLMs

# 💻 Code Completion
- Full app generation from natural specs  
- Security flaw detection & fixes  
- Custom automation scripts (SQL, Python, RPA)  

---

## Concrete Applications of LLMs

# 🤖 Chatbots & Assistants
- 24/7 customer support (80–90% automated)  
- Smart personal assistants (scheduling, admin, comparisons)  
- Healthcare support: symptom pre-analysis, treatment reminders  

---

## Concrete Applications of LLMs

<div style="display: flex; align-items: center; gap: 20px;">

  <div style="flex: 1;">

# 🌍 Other Applications
- Context-aware translations (legal, cultural)  
- Document analysis & insights extraction  
- Adaptive tutoring & personalized learning  
- Business workflows: meeting summaries, decision tracking  

</div>

  <div style="flex: 1;">

<img src="https://media.licdn.com/dms/image/v2/D4D12AQHpYvUjqjFIgg/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1738606568800?e=1760572800&v=beta&t=kx2JCU6pebZi9M6tBZXI8WiSWZ8DNZfnnFUAksnu1YI" alt="LLMs Overview" style="width:100%;">

</div>
---

# Attention Mechanism: Enhancing Neural Networks

<div style="display: flex; align-items: center; gap: 20px;">

  <div style="flex: 1;">

## Publication 2017
- **2017:** Vaswani et al. proposed **"Attention is All You Need,"** introducing the Transformer architecture.
- **Impact:** Became the foundation for modern LLMs like GPT and BERT, replacing RNNs in many applications.


## Key Features
- **Selective Focus:** Assigns weights to input elements, emphasizing the most relevant parts.
- **Interpretability:** Highlights which parts of the input influence the output, aiding in model transparency.

<!--  -->
## Applications
- **Natural Language Processing:** Machine translation, summarization, and question answering.
- **Computer Vision:** Image captioning and object detection.
- **Automotive Industry:** Predicting brake fade, analyzing driver behavior, and optimizing ADAS systems.

</div>

  <div style="flex: 1;">

<div style="text-align: center;">
  <img src="https://0.academia-photos.com/attachment_thumbnails/84202720/mini_magick20220415-14619-1w85ue4.png" alt="Attention Mechanism" style="width: 50%;">
</div>

  </div>

</div>

---

# Attention Mechanism Equation

The attention mechanism can be mathematically expressed as:

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

Where:
- $Q$ (Query): What we're looking for
- $K$ (Key): What we're comparing against  
- $V$ (Value): The actual information to retrieve
- $d_k$: dimension of the key vectors (used for the scaling factor)

This mechanism allows the model to **focus on the most relevant parts** of the input sequence.

---

# Why **Attention** can be a game changer
- Captures very long-term dependencies without the memory degradation typical of RNNs.  
- Produces interpretable attention maps: helps identify which past steps influence the current prediction.  
- Enables modelling of rare but critical events by directly linking distant cues in the sequence.

---
# **Attention** and Predictive maintenance (sensor data, time series)  
- Temporal self-attention on sensor logs → early detection of anomalies and progressive wear.  
- Attention maps: temporal localization of root causes (e.g., vibration spikes preceding failure).  

---

# Attention and Modeling brake fading / friction  

<div style="display: flex; align-items: center; gap: 20px;">

  <div style="flex: 1;">

- Causal temporal transformer → identifies stops/thermal events that lead to friction drops.  
- Multi-head causal attention: each head captures different time scales 
  - short: thermal spikes 
  - long: accumulated energy

  </div>

  <div style="flex: 1;">  

![Types of Brake Fade](https://www.slashgear.com/img/gallery/what-is-brake-fade-and-what-causes-it/what-are-the-types-of-brake-fade-1707264185.jpg)

  </div>
</div>
---

## Attention Mechanism and Detection of critical events in long test campaigns  

- Automatic spotting of significant epochs (hard stops, long heat-ups) to prioritize HIL / bench validation.

---

# Quiz: Attention Mechanism

---

# Why is the attention mechanism crucial in modern neural networks?
  - A) It replaces the need for activation functions.
  - B) It allows the model to focus on the most relevant parts of the input sequence.
  - C) It eliminates the need for training data.
  - D) It reduces the size of the neural network.

---

# <div style="display: flex; justify-content: center; align-items: center; height: 100vh; text-align: center; font-size: 10vw; font-weight: bold; width: 100%;">Fine-tuning</div>

---

# Full Training

Training a large language model or neural network from scratch is computationally expensive and resource-intensive. It requires:

- **Massive Datasets:** Billions of tokens across diverse domains.
- **High Compute Power:** Specialized hardware like GPUs/TPUs and significant energy consumption.
- **Time:** Training can take weeks or months, even on large-scale infrastructure.

---
# Fine-Tuning as a Solution

Fine-tuning leverages pre-trained models and adapts them to specific tasks or domains. This approach:

- **Reduces Costs:** Requires significantly less compute and time compared to full training.
- **Improves Performance:** Tailors the model to domain-specific data, enhancing accuracy and relevance.
- **Increases Accessibility:** Enables smaller teams to build specialized applications without extensive resources.

---
# Fine-Tuning

**Example:**
- Fine-tuning GPT or Mistral on automotive datasets (e.g., technical manuals, sensor logs) can create a specialized model for predictive maintenance or driver assistance systems at a fraction of the cost of full training.

---
# Example on French Gastronomy

## Pre-training
The base model (e.g., GPT or Mistral) is trained on a **general corpus**:  
- Web articles, books, forums, Wikipedia, various recipes…  
- It already understands French, sentence structures, common ingredients, etc.  
- However, it **is not an expert** in French gastronomy or precise chef techniques.


---
## Example on French Gastronomy

<div style="display: flex; align-items: center; gap: 20px;">

  <div style="flex: 1;">

# Fine-tuning
The training continues on a **targeted dataset**:  
- French gastronomic recipes: classic sauces, cooking techniques, refined flavor pairings.  
- Chef notes, gastronomy books, Michelin-starred menus…  

  </div>

  <div style="flex: 1;">
```mermaid
flowchart TD
  A[General Pre-trained Model] --> |General Knowledge| B[Fine-tuning on French Gastronomic Recipes]
  B --> |Learning Techniques and Style| C[Specialized Model in French Gastronomy]
```
  </div>
</div>

---

## Example on French Gastronomy

<div style="display: flex; align-items: center; gap: 20px;">

  <div style="flex: 1;">

# The model learns to:  
- Recognize specific terms: “sauce bordelaise,” “low-temperature cooking,” “déglaçage au vin rouge”
- Suggest ingredients and techniques that are more **authentic** to French cuisine.  
- Adhere to the **gastronomic and precise** style of Michelin-starred recipes.

  </div>

  <div style="flex: 1;"> 

  <img src="https://s.france24.com/media/display/8d788430-1f7c-11e9-852c-005056bff430/w:1280/p:16x9/dominique_crenn_-_m.jpg" alt="Dominique Crenn" style="width: 100%;">

  </div>

---
# Example on French Gastronomy

<div style="display: flex; align-items: center; gap: 20px;">

  <div style="flex: 1;">

  ## Result
  After fine-tuning, the model can:  
  - Generate **complete and realistic French gastronomic recipes**.  
  - Adapt existing recipes into a **gastronomic style**.  
  - Answer questions such as:  
    > “How to prepare sweetbreads with morels and red wine sauce like a Michelin-starred chef?”
  </div>
  <div style="flex: 1;">

![Image](https://cdn.sanity.io/images/d35hevy9/production/45a0d7d20ff15bbf7f88654b848379b9fd0b9d10-2756x1480.png)

  </div>
 </div> 

---

# Quiz: Fine-Tuning

---

# What is the primary benefit of fine-tuning a pre-trained AI model?
  - A) It eliminates the need for training data.
  - B) It reduces the computational cost compared to training from scratch.
  - C) It increases the size of the model.
  - D) It replaces the pre-trained model with a new one.

---

# AI on Cloud, Sovereign, On-Premise, or Edge

---

# AI on Cloud

## Description: AI is hosted on public cloud platforms like AWS, Azure, or Google Cloud.
- **Advantages:**
  - Rapid scalability.
  - Easy access to pre-trained models and massive resources.
  - *Low initial cost.*
- **Disadvantages:**
  - Data privacy concerns.
  - Dependence on internet connectivity.
  - Risk of vendor lock-in.

---
# AI on Cloud

- **Examples:**
  - **AWS SageMaker** for training and deploying ML models.
  - **Google Vertex AI** for integrated AI solutions.
  - **Microsoft Azure AI** for cognitive services.

---
# Sovereign AI

<div style="display: flex; align-items: center; gap: 20px;">

  <div style="flex: 1;">

- **Description:** Infrastructure and data are managed within a country to comply with local regulations.
- **Advantages:**
  - Full control over data.
  - Compliance with national regulations.
  - Enhanced national security.
- **Disadvantages:**
  - High setup cost.
  - Limited scalability compared to global cloud.
  - Requires internal maintenance.

  </div>
  <div style="flex: 1;">

![Image](https://media.securiti.ai/wp-content/uploads/2025/06/11045706/Securiti-Powers-Sovereign-AI-in-the-EU-with-NVIDIA.png)

  </div>
</div>

---
# Sovereign AI

## Typical Use Cases: Government projects, defense, healthcare requiring strict data residency.
- **Examples:**
  - **Gaia-X:** European initiative for a sovereign cloud.
  - **OVHcloud:** French provider of sovereign cloud solutions.


---
# AI On-Prem

<div style="display: flex; align-items: center; gap: 20px;">

  <div style="flex: 1;">

- **Description:** AI is deployed on the organization’s own servers and data centers.
- **Advantages:**
  - Full control over hardware and data.
  - Low latency for local users.
  - Tight integration with internal systems.
- **Disadvantages:**
  - High upfront investment. (Nvidia H100 ~ $30 000 each)
  - Requires in-house expertise.
  - Scaling is expensive.

  **ask your IT guy**

  </div>
  <div style="flex: 1;">
  
![Image](https://miro.medium.com/v2/resize:fit:970/1*TFadQrVWmT6FsYlIG5HDnw.jpeg)
  
  </div>
</div>


---
# AI On-Prem
- **Typical Use Cases:** Internal analytics, finance, R&D labs.
- **Examples:**
  - **NVIDIA DGX Systems** for on-site AI model training.

<div style="text-align: center;">
  <img src="https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/dgx-a100/data-denter-dgx-a100-tour-1cN-D@2x.jpg" alt="DGX Systems" style="width: 50%;">
</div>
---

# AI on Edge

<div style="display: flex; align-items: center; gap: 20px;">

  <div style="flex: 1;">

- **Description:** AI is deployed on devices close to the data source (IoT, mobile, industrial machines).
- **Advantages:**
  - Ultra-low latency.
  - Reduced bandwidth usage.
  - Can operate offline.
- **Disadvantages:**
  - Limited computing resources.
  - Model updates are more complex.
  - Security of distributed devices.


  </div>

  <div style="flex: 1;">

![nvidia Robot](https://developer-blogs.nvidia.com/wp-content/uploads/2024/10/humanoid-robot-gif.gif)

  </div>
</div>

---

# AI on Edge

<div style="display: flex; align-items: center; gap: 20px;">

  <div style="flex: 1;">

- **Typical Use Cases:** Autonomous vehicles, smart cameras, industrial automation, IoT devices.

- **Examples:**
  - **Qualcomm Snapdragon Ride** for autonomous vehicles.
  - **NVIDIA Jetson** for IoT and robotics applications.

</div>

  <div style="flex: 1;">

![Snapdragon Ride](https://specials-images.forbesimg.com/imageserve/61d625f9b6cd3ef368034eac/All-the-vehicle-system-included-in-the-Snapdragon-Digital-Chassis-and-current/960x0.jpg?fit=scale)

  </div>
</div>

---
# AI deployment strategies in a wrap

AI deployment strategies should align with specific automotive use cases, balancing performance, cost, and regulatory compliance.

| AI Strategy | Description | Advantages | Challenges | Typical Use Cases |
|-------------|-------------|------------|------------|-----------------|
| **Cloud** | AI services hosted on public cloud platforms (AWS, Azure, GCP). | - Scalability<br>- Easy access to large models<br>- Low upfront cost | - Data privacy concerns<br>- Dependence on internet connectivity<br>- Possible vendor lock-in | Chatbots, recommendation systems, analytics, SaaS AI solutions |
| **Sovereign / National** | AI infrastructure and data managed within a country to meet regulatory and sovereignty requirements. | - Full control over data<br>- Compliance with local regulations<br>- Enhanced national security | - High setup cost<br>- Limited scalability compared to global cloud<br>- Maintenance responsibility | Government AI projects, defense, healthcare requiring strict data residency |
| **On-Premises (On-Prem)** | AI deployed on the organization’s own servers and data centers. | - Full control over hardware and data<br>- Low latency for local users<br>- Can integrate tightly with internal systems | - High upfront investment<br>- Requires in-house expertise<br>- Scaling can be slow and expensive | Sensitive enterprise AI, internal analytics, finance, R&D labs |
| **Edge** | AI deployed on devices close to the data source (IoT, mobile, industrial machines). | - Ultra-low latency<br>- Reduced bandwidth usage<br>- Can operate offline | - Limited computing resources<br>- Model updates more complex<br>- Security of distributed devices | Autonomous vehicles, smart cameras, industrial automation, IoT devices |


---

# Quiz: AI Deployment Strategies

---

# Which AI deployment strategy is ideal for autonomous vehicles and IoT devices?
- A) Cloud
- B) Sovereign / National
- C) On-Premises
- D) Edge

---

# What is a primary challenge of on-premises AI deployment?
- A) Dependence on internet connectivity
- B) Limited computing resources
- C) High upfront investment and in-house expertise requirements
- D) Vendor lock-in

---

# <div style="display: flex; justify-content: center; align-items: center; height: 100vh; text-align: center; font-size: 10vw; font-weight: bold; width: 100%;">AI ACT and EU Regulations</div>

---

# AI Risk Categories and Regulations

| Risk Category            | Description                                                                                     | Examples                          | Measures/Recommendations                                                                 |
|--------------------------|-------------------------------------------------------------------------------------------------|-----------------------------------|-----------------------------------------------------------------------------------------|
| Unacceptable Risk        | Systems prohibited as they pose a serious threat to fundamental rights or safety.               | Social scoring, Psychological manipulation, Unregulated biometric recognition         | Strictly prohibited                                                                     |
| High Risk                | Systems with a significant impact on safety or fundamental rights.                              | Medical diagnosis, AI for recruitment, Management of critical infrastructures          | Conformity assessments, Technical documentation, Mandatory human oversight              |
| Limited Risk             | Systems with moderate risk requiring transparency obligations.                                   | Chatbots, AI-generated image/voice filters                                            | Inform users that they are interacting with AI or that content is artificially generated |
| Minimal Risk             | Systems with low risk, without specific regulatory obligations.                                  | Recommendation systems, AI in video games                                             | Encourage voluntary codes of conduct to improve transparency and fairness               |

---

# <div style="display: flex; justify-content: center; align-items: center; height: 100vh; text-align: center; font-size: 10vw; font-weight: bold; width: 100%;">Foundation Models: The Backbone of AI Innovation</div>

---

# Foundation Models

**Definition:**
- Foundation models are **pre-trained, versatile, and powerful AI models** capable of generalizing across many tasks and domains.
- They serve as a base for developing specific applications via fine-tuning or prompt engineering.

**Examples:**
- **GPT-5 (OpenAI):** Advanced language model for text generation, context understanding, and complex problem-solving.
- **LLaMA (Meta):** Open-source large model adaptable to various use cases.
- **Mistral 7B (Mistral AI):** French state-of-the-art model optimized for natural language understanding and generation.
- **Kyutai :** Specialized in natural language processing and speech recognition, with strong multilingual capabilities.
- **Whisper (OpenAI):** Robust speech-to-text model supporting multiple languages and optimized for noisy environments.


**Importance:**
- Reduce development costs and time.
- Offer superior generalization capabilities.
- Adaptable to diverse domains, including automotive.

---

![Foundation Models Explained](https://humanloop.com/blog/foundation-models/foundation-models-explained.jpeg)

---
# Foundation Models

| **Domain** | **Foundation Models (examples)** |
|------------|-----------------------------------|
| **Text (LLMs)** | GPT-4 / 5 (OpenAI), Claude 3.5 (Anthropic), Gemini 1.5 (Google), LLaMA 3 (Meta), Mistral / Mixtral / Codestral, Command-R+ (Cohere), Jamba (AI21 Labs), Grok (xAI) |
| **Vision** | CLIP, DALL·E 3 (OpenAI), Stable Diffusion SDXL / SD3 (Stability AI), Imagen, Flamingo (Google), Segment Anything (Meta), DINOv2 |
| **Multimodal (Text + Image + Audio + Video)** | GPT-4o (OpenAI), Gemini 1.5 (Google), Claude 3.5 multimodal (Anthropic), Kosmos-2 (Microsoft), LLaVA, IDEFICS (Hugging Face) |
| **Audio & Speech** | Whisper (OpenAI), VALL-E (Microsoft), SeamlessM4T, MMS (Meta), ElevenLabs Voice AI, Bark, XTTS |
| **Code & Reasoning** | Codestral (Mistral), DeepSeek-Coder, CodeLLaMA (Meta), GPT-o1 reasoning (OpenAI), StarCoder (Hugging Face + ServiceNow) |
| **Agents / RAG Frameworks** | LangChain, LlamaIndex, Haystack (not models but key for orchestration) |

---

# Benefits of Foundation Models

**Cost and Time Reduction:**
- Reduce development time and cost by **50–80%**.
- Eliminate the need to build specialized models from scratch.

**Performance Improvement:**
- Superior accuracy and robustness compared to specialized models.
- Ability to generalize across multiple tasks and domains.

---

# Benefits of Foundation Models

**Adaptability and Customization:**
- **Fine-tuning** for specific applications.
- Adaptation to languages, accents, and contexts.

**Accelerated Innovation:**
- Rapid launch of new features and services.
- Facilitate the integration of new technologies.

---

# <div style="display: flex; justify-content: center; align-items: center; height: 100vh; text-align: center; font-size: 10vw; font-weight: bold; width: 100%;">Computer Vision & Multimodality</div>

---
# YOLO

## **Y**ou
## **O**nly
## **L**ook
## **O**nce
---


## Object Detection (YOLO/R-CNN)
![Object Detection](https://cv-tricks.com/wp-content/uploads/2017/12/Object-Detection-for-outdoor-cv-tricks.jpg)

---

# Goal: Identify and Locate Objects in an Image

- **Objective:** Identify and locate objects within an image by providing their class and bounding box.
- **Applications in Automotive:**
  - Pedestrian and vehicle detection for autonomous driving.
  - Traffic sign recognition.
  - Obstacle detection and avoidance.
  - Parking assistance systems.

---

# Two Main Approaches

## R-CNN Family
- **Process:**
  - Generates region proposals in the image.
  - Classifies each region using a convolutional neural network.
- **Advantages:**
  - High accuracy due to region-based processing.
  - Effective for detecting small objects.
- **Disadvantages:**
  - Computationally expensive.
  - Slower processing speed.

---

# R-CNN Family

- **Example:**
  - **Mobileye** has developed camera-only Intelligent Speed Assist solutions for **traffic sign recognition**, based on deep convolutional networks (approaches related to region-based object detection).
    - [Mobileye Blog, 2024](https://www.mobileye.com/blog/intelligent-speed-assist-isa-computer-vision-adas-solution/)
  - **Bosch** has released research datasets (e.g., Bosch Small Traffic Lights Dataset) used with **Faster R-CNN and Mask R-CNN** in academic and industrial collaborations for detecting small objects like traffic lights.
    - [Bosch Small Traffic Lights Dataset, GitHub](https://hci.iwr.uni-heidelberg.de/node/6132)
  - **Academic/industrial joint research** (e.g., Shao et al., 2019, Applied Sciences) applied **Improved Faster R-CNN** for traffic sign detection, achieving higher accuracy on small signs in real-world driving datasets.
    - [Applied Sciences, MDPI 2019](https://doi.org/10.3390/app12188948)

---

# YOLO Family

<div style="display: flex; align-items: center; gap: 20px;">

  <div style="flex: 1;">

- **Process:**
  - Single-shot detection: Predicts classes and bounding boxes in one pass through the network.

- **Variants:**
  - YOLOv3, YOLOv4, YOLOv5, YOLOv6, YOLOv7, YOLOv8

- **Advantages:**
  - Real-time speed.
  - Efficient and fast detection suitable for real-time applications.

- **Disadvantages:**
  - Slightly lower accuracy compared to R-CNN.

  </div>

  <div style="flex: 1;">

<div style="text-align: center;">
  <img src="https://framerusercontent.com/images/wmnihLAMWTEv1yW0v6w4ETUz3BQ.jpeg" alt="YOLO" style="width: 75%;">
</div>

  </div>

</div>

---

# YOLO in action

Hitachi Astemo Winter Test - Smart Brake with YOLO
<iframe width="640" height="480" src="https://www.youtube.com/embed/8FkGvBHhfcI" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---

# 🎯 YOLO Live Demo

**Instructions :** Appuyez sur [**ESPACE**] pour Démarrer/Arrêter YOLO

---

<iframe src="Labs/yolo_live_stream.html?autostart=false" 
        style="width: 100vw; height: 100vh; border: none; margin: 0; padding: 0;"
        allowfullscreen>
</iframe>


<style>
@keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0.3; }
}
</style>

---






# YOLO Use-Cases in Automotive / Traffic Domain
| Use Case | YOLO Variant / Approach | Description | Source |
|----------|--------------------------|-------------|--------|
| Traffic sign detection with YOLOv8 | YOLOv8-based algorithm (YOLO-BS) | Improved traffic sign detection with high mAP on public datasets. | [“A traffic sign detection algorithm based on YOLOv8” — 2025, H. Zhang et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11880478/) |
| Context-based sign detection with YOLOv7 | YOLO-CCA (YOLOv7 + context modules) | Adds local/global context modules and Transformer-based feature fusion, boosting mAP by ~3.9% on Tsinghua-Tencent-100K. | [YOLO-CCA: A Context-Based Approach for Traffic Sign Detection](https://arxiv.org/abs/2412.04289) |
| YOLO in traffic sign detection: A review | Multiple YOLO variants (v1-v8) | Literature review (2016–2022) on YOLO for traffic signs, datasets, hardware, metrics, and challenges. | [Traffic Sign Detection and Recognition Using YOLO Object Detection Algorithm: A Systematic Review (Mathematics 2024)](https://www.mdpi.com/2227-7390/12/2/297) |
| Small traffic sign detection | TRD-YOLO | Optimized for small traffic signs, improving accuracy on small objects. | [TRD-YOLO: A Real-Time, High-Performance Small Traffic Sign Detection Algorithm (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10145582/) |
| Pedestrian detection with attention modules | FA-YOLO (YOLO + Feature Enhancement + Adaptive Sparse Self-Attention) | Enhanced pedestrian detection with robustness to occlusion and lighting variations. | [FA-YOLO: A Pedestrian Detection Algorithm with Feature Enhancement … (Electronics 2025)](https://doi.org/10.3390/electronics14091713) |


---

## YOLO Use-Cases in Automotive / Traffic Domain

# Video: Object Detection in Action

  <iframe width=640 height=480 src="https://www.youtube.com/embed/FdZvMoP0dRU?autoplay=1" title="Reinforcement Learning in Action" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen style="font-size: 1.5em;"></iframe>

---

# Trade-off: Accuracy (R-CNN) vs Real-time Speed (YOLO)

| **Aspect**            | **R-CNN Family**                     | **YOLO Family**                      |
|-----------------------|---------------------------------------|---------------------------------------|
| **Accuracy**          | High (better for small objects)      | Moderate                            |
| **Speed**             | Slower (multi-stage process)         | Faster (real-time capable)          |
| **Computational Load**| Higher                                | Lower                               |
| **Real-time Use**     | Limited                               | Suitable                            |

---

# Computer Vision Use Cases in Automotive

| Automaker        | Use Case                                                                 | Vision-centric Feature(s) / Functions                                  | Source (URL)                                                                 |
|------------------|--------------------------------------------------------------------------|------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| **Tesla**        | FSD v12.5: Vision-based attention monitoring (cabin camera), plus améliorations dans le monitoring avec lunettes de soleil etc. | Camera-only driver attention monitoring, suppression des alertes manuelles (“nags”), Vision-based attention system | [Electrek: FSD v12.5 first impression](https://electrek.co/2024/09/24/tesla-full-self-driving-v12-5-first-impression-more-natural-dangerous/)  |
| **Mercedes-Benz** | CLA (2025) to offer “hands-off point-to-point autonomous urban driving capability” in some jurisdictions | ADAS / driving assistance urbain, likely vision + sensor fusion pour fonctionnement mains-libres dans trafic urbain                      | [WardsAuto: 2025 Mercedes-Benz CLA to Offer Autonomous Urban Driving Capability](https://www.wardsauto.com/autonomous-adas/2025-mercedes-benz-cla-to-offer-autonomous-urban-driving-capability)  |
| **Toyota**        | Next-gen vehicles to be built with NVIDIA Drive AGX Orin supercomputer + DriveOS for automated driving capabilities (announced at CES 2025) | Real-time computer vision + sensor fusion via Nvidia Drive AGX Orin; implies visual perception & ADAS/autonomous driving support | [Toyota & NVidia](https://techstory.in/toyota-teams-up-with-nvidia-for-next-gen-automated-driving-at-ces-2025/)  |
| **Mercedes-Benz** | Partnership with Momenta: Mercedes to use Momenta software on at least four models in China from 2025-2027 for autonomous driving/ADAS features | Use of vision-based ADAS / autonomy features supplied by Momenta; likely includes perception modules | [Reuters: Mercedes to use Momenta software in 4 models](https://www.reuters.com/business/autos-transportation/mercedes-use-momenta-software-4-models-accelerate-china-comeback-sources-say-2024-11-29/)  |


---
## Tesla Vision-based Driver Monitoring

<video controls autoplay width="640" height="480">
  <source src="https://digitalassets.tesla.com/tesla-contents/video/upload/f_auto,q_auto/network.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


---

# <div style="display: flex; justify-content: center; align-items: center; height: 100vh; text-align: center; font-size: 10vw; font-weight: bold; width: 100%;">Voice AI</div>

---

# Voice Assistants

**Functionalities:**
- Hands-free control of navigation, climate, infotainment, and third-party apps.
- Integration with embedded systems (e.g., MBUX, BMW iDrive, Tesla OS).

**Industrial Examples:**
- **Mercedes-Benz "Hey Mercedes":** Understands natural commands ("I'm cold" → automatic temperature adjustment).

---

# TTS (Text-to-Speech)

**Technologies:**
- Neural models (e.g., Amazon Polly, Google WaveNet, ElevenLabs, Kyutai TTS) for human-like speech.
- Customization for brand identity.

**Use Cases:**
- Safety alerts ("Attention, vehicle in blind spot") with urgent but calm intonation.
- Conversational feedback ("Your destination is 10 minutes away. Would you like a coffee break?").

---

# STT (Speech-to-Text)

**Leading Models:**
- **[OpenAI Whisper](https://openai.com/research/whisper):** Robust to noise, supports 99 languages, offline capable.
- **[Kyutai STT](https://kyutai.org/):** French state-of-the-art model, optimized for noisy and multilingual environments.

**Performance:**
- Whisper (large-v3): 98% accuracy on English.


---

# Speaker Diarization

**Technologies:**
- Models like PyAnnote to identify speakers.

**Use Cases:**
- Uses diarization to distinguish between driver and passengers.
---

# Benefits of Voice Technologies

| **Benefit**               | **Impact for Driver**               | **Impact for OEM**              |
|---------------------------|------------------------------------|--------------------------------|
| Hands-free control        | Reduced distractions, improved safety | Compliance with regulations (e.g., Euro NCAP) |
| Safety                    | Fewer accidents related to screen manipulation | Lower warranty/recall costs |
| Personalization           | Adapted experience (voice, preferences, history) | Customer loyalty and brand differentiation |
| Accessibility             | Usable by all (including visually impaired) | Expanded market (e.g., seniors) |
| Multilingual support      | Commands in local language          | Global sales without costly adaptations |
| Context awareness         | Adapted responses (e.g., "Let's go home" → automatic navigation) | Data to improve connected services |

**Sources:**
- [McKinsey, 2023](https://www.mckinsey.com/industries/automotive-and-assembly/our-insights)

---

# Challenges & Technical Solutions

| **Challenge**              | **Solution**                                                                 |
|----------------------------|------------------------------------------------------------------------------|
| Background noise (engine, music) | Whisper + signal processing (e.g., NVIDIA NSA filters)                      |
| Regional accents/languages  | Fine-tuning Whisper or Kyutai on local datasets (e.g., Quebec French)       |
| Latency                    | On-device models (e.g., Whisper Tiny on Snapdragon Ride)                    |
| Privacy                    | Local processing (no cloud) and voice data anonymization                    |
| Multi-system integration   | Unified platforms (e.g., Android Automotive)                           |

---

# Typical Voice Interaction Workflow

- **Audio Capture** (microphone array).
- **Signal Cleaning** (noise/music suppression).
- **STT** (e.g., Whisper or Kyutai) → Text.
- **Diarization** (who is speaking? driver or passenger?).
- **NLP** (intent understanding).
- **Action** (e.g., change temperature).
- **TTS** (natural voice response).

---

# Recommendations for Directors

- Start with a pilot using Whisper or Kyutai (offline) + basic TTS for critical commands (navigation, climate).
- Integrate diarization for family vehicles (e.g., SUVs).
- Customize voice for brand identity.
- Plan for multilingual support from day one.
- Ensure privacy by processing voice data locally.

---

# Future Trends

- **Voice Biometrics:** Driver recognition via voice.
- **Emotion AI:** Stress or fatigue detection via voice analysis.
- **Multimodal:** Combining voice + gestures + gaze (e.g., "Open the window" + looking at the window).
- **Generative AI:** Using embedded LLMs (e.g., Mistral 7B, Kyutai) for open-ended conversations ("Why is my fuel consumption high today?").

---

# Arguments to Convince Decision Makers

- **Safety:** "Voice user interface: a complementary safety solution"
- **Differentiation:** "A smooth voice interface is a **key selling point** for young drivers (Gen Z/Y)."
- **ROI:** "A well-designed voice system can **increase margins by 5–10%** through premium options."
- **Regulation:** Matthew Avery, director of strategic development for Euro NCAP, recently said, “The overuse of touchscreens is an industry-wide problem, with almost every vehicle maker moving key controls onto central touchscreens, obliging drivers to take their eyes off the road, and raising the risk of distraction crashes.”.

---

# Key Points to Remember

- **Core Message:** Voice is a **natural and safe** interface for car interaction.
- **Impactful Numbers:** 98% accuracy with Whisper
- **Call to Action:** "Launch a pilot with **Whisper or Kyutai + customized TTS**

---

# Demo with unmute.sh

- [unmute.sh](https://unmute.sh/)
- **Real-time Interaction:** Voice interaction without perceptible delay, more natural.
- **Configurable Voices:** From short samples, without heavy training.
- **Behavioral Personalization:** Via simple text prompts.
- **Deployment Flexibility:** On existing systems, no cloud dependency.
- **Immediate Testing:** Code publication imminent.

---

<div style="width: 100%; height: 600px; overflow: hidden; border: 1px solid #ccc;">
  <iframe
    src="https://unmute.sh/"
    width="100%"
    height="650px"
    frameborder="0"
    style="border: none; overflow: hidden;"
    scrolling="no">
  </iframe>
</div>

<div style="text-align: center; margin: 2em;">
    <button
      onclick="window.open('https://unmute.sh/', '_blank')"
      style="
        padding: 14px 28px;
        background: #4CAF50;
        color: white;
        border: none;
        font-size: 20px;
        border-radius: 6px;
        cursor: pointer;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        transition: background 0.3s;
      "
      onmouseover="this.style.background='#45a049'"
      onmouseout="this.style.background='#4CAF50'">
      Ouvrir unmute.sh dans un nouvel onglet
    </button>
  </div>


<!-- Here’s your **complete, ready-to-use prompt in English** for an EV’s AI companion, optimized for a foggy 2-hour drive with technical precision and natural interaction:

**You are the AI companion of an electric vehicle (e.g., Tesla Model 3 or Hyundai Ioniq 5), blending deep technical expertise in EVs, embedded systems, and AI with the empathy of an attentive co-pilot. Your role is to ensure a **safe, efficient, and enjoyable** 2-hour journey in foggy conditions (4°C, 92% humidity, 50m visibility), while adapting to the driver’s preferences and needs.**

### **Trip Context (Real-Time Data)**
- **Route:** 187 km remaining (2h estimated arrival at **20:00**).
- **Time:** 18:00 (sunset at 18:47).
- **Weather:** Dense fog (visibility ~50m), light traffic with trucks every 3–5 km.
- **Traffic Alert:** Slowdown near Orléans (km 45, +12 min historical delay). Alternate route via A71 adds 6 km but avoids trucks.
- **Next Stop:** **Chartres Service Area (15 km ahead)** – 4x 150 kW chargers (2 available), coffee (⭐4.2), restrooms, free washer fluid refill.

### **Vehicle Status**
| **Parameter**               | **Value**                     | **Notes**                                                                 |
|-----------------------------|-------------------------------|---------------------------------------------------------------------------|
| **Battery SoC**             | 78% (280 km range)            | Sufficient for destination + 50 km buffer.                               |
| **Battery Temp**            | 22°C                          | Optimal (15–30°C). Preconditioning active for efficiency.                |
| **Tire Pressure**           | Front: 2.4 bar and Rear: 2.5 bar | Cold-weather optimal (spec: 2.3–2.6 bar).                                |
| **Tire Temp**               | 10°C (all)                    | Normal for winter; grip monitoring enabled.                              |
| **Energy Consumption**      | 18.2 kWh/100 km               | Slightly high due to fog (heating/lights). Average: 16.5 kWh/100 km.      |
| **Regen Braking**           | 65% efficiency                | Reduced by wet roads; coasting recommended.                              |
| **Climate Control**         | Cabin: 21°C, Seat Heat: Lv2   | Using 1.2 kW/h. Suggest reducing to Lv1 to save 0.5 kW/h.                 |
| **Defrost/Fog Lights**      | ON (1.5 kW/h)                 | Auto-adjusts based on humidity sensors.                                  |
| **Headlights**              | Low beam + fog lights         | Auto-high beam disabled due to fog.                                       |
| **Washer Fluid**            | 38%                           | Low—suggest refill at Chartres.                                           |
| **Safety Systems**          | ACC (85 km/h, 3-sec follow), LKA active | Last nudge: 5 km ago (minor drift).                        |
| **Air Quality**             | PM2.5: 6 µg/m³                | Excellent (HEPA filter active).                                           |

### **Driver Profile**
- **Interests:** Convolutional neural networks (CNNs), virtual sensors, *Kreyol Keyboard* project, electronic music.
- **Recent Activity:** Listened to *Lex Fridman Podcast #347* (edge AI).
- **Preferences:** Tech discussions, Daft Punk/Hans Zimmer music, highly rated coffee stops.
- **Fatigue Level:** Low (HR: 68 BPM, no yawning).

### **Interaction Guidelines**
1. **First Interaction (18:00):**
   - *"How can I help you?"*
   - **If no response after 5 sec:**
     *"Here’s what I’m optimizing for you:
     - **Fog Adaptations:** Reduced speed to **78 km/h** in dense areas (range impact: -5 km). Defrost at Lv3 (1.5 kW/h).
     - **Tech Topic:** NVIDIA’s new *CNN for low-light sensor fusion* (2024) could inspire your virtual sensor work—2-minute summary?
     - **Comfort:** Seat at 21°C. Lower to 19°C to save 0.3 kW/h, or play *Hans Zimmer’s ‘Time’*?
     - **Practical:** Chartres rest stop in 15 km has open chargers. Need a 5-minute coffee break? I’ll pre-condition the battery for faster charging."*

2. **During the Trip:**
   - **If consumption >19 kWh/100 km:** *"Cabin heat and lights add 2 kWh/h. ‘Chill Mode’ could save 3% range. Shall I adjust?"*
   - **If tire pressure ↓0.3 bar:** *"Left rear tire at 2.1 bar (cold). Monitoring; inflate at Chartres if <2.0 bar."*
   - **If fatigue detected:** *"Your blink rate’s up. Chartres in 10 km has coffee. Or I can explain how Tesla’s fog-light CNN works—your call!"*

3. **Proactive Alerts:**
   - **Fog:** *"Visibility drops to 30m in 8 km. Slowing to 70 km/h, increasing defrost. Trucks ahead—maintaining 4-sec distance."*
   - **Range:** *"280 km range (103% of needed). Cold reduces efficiency by 8%. Plug in at home for 80% charge (optimal for battery)."*
   - **Tech Curiosity:** *"Modern CNNs use *denoising layers* to improve pedestrian detection in fog—like your virtual sensor project!"*

4. **Approaching Destination (19:30):**
   - *"Recap: 210 km range left (103% of trip). No incidents despite 23 km in dense fog.
     - **For tomorrow:** Frost forecasted. Shall I pre-condition the battery at 6 AM (5% range boost)?"*

5. **Emergencies:**
   - **If SoC <30%:** *"Orléans Supercharger (45 km) has 3 open stalls. 10-minute charge adds 120 km. Stop?"*
   - **If sudden slowdown:** *"Truck accident ahead (5 km). Rerouting via A71 (+6 km, clear traffic). Decide?"*

### **Tone & Style**
- **Technical but accessible:** *"Your battery’s at 22°C—ideal for efficiency. Below 10°C, range drops 20%. That’s why I pre-warmed it during charging."*
- **Empathetic & reassuring:** *"Fog can be stressful, but we’ve 50 km buffer and sensors monitoring 250m around us. I’ll alert you if action’s needed."*
- **Humorous (if profile fits):** *"Fog lights use 200W—0.0003 Daft Punk songs per km. A necessary sacrifice for safety!"*

### **Rules**
- **Language:** Match driver’s language (French/English).
- **Proactive:** *"The Chartres stop has free washer fluid—shall I add it to our to-do list?"*
- **Transparent:** Explain auto-decisions (e.g., *"Reduced regen to 60% for wet roads—protects tires and battery."*).

**Customization Options:**
- Add **EV-specific tips** (e.g., *"Your Ioniq 5’s heat pump is 30% more efficient than resistive heating—perfect for this weather."*).
- Reference **driver’s projects** (e.g., *"Like your Kreyol Keyboard, CNNs balance efficiency and accuracy!"*). -->

---

# Future Trends

- **Multimodal Foundation Models:** Integration of voice, text, images, and sensor data for enhanced contextual understanding.
- **Embedded Models:** Deployment on specialized chips (e.g., Qualcomm Snapdragon, NVIDIA Orin) for reduced latency and improved privacy.
- **Advanced Personalization:** Tailoring models to individual driver preferences (e.g., voice, language, interaction style).
- **ADAS Integration:** Using foundation models to improve contextual understanding and decision-making in advanced driver assistance systems.

---

# Arguments to Convince Decision Makers

- **Innovation and Competitiveness:**
  - "Foundation models are **key** to staying competitive in the AI and autonomous vehicle era."

- **Cost Reduction:**
  - "Reduce development costs by 50–80% while improving application quality and robustness."

- **Customer Satisfaction:**
  - "Enhance customer satisfaction with natural and personalized voice interactions."

- **Regulatory Compliance:**
  - "Meet regulatory requirements for safety and accessibility."

---

# # <div style="display: flex; justify-content: center; align-items: center; height: 100vh; text-align: center; font-size: 10vw; font-weight: bold; width: 100%;">AI Applied to the Automotive Industry</div>

---

# Stellantis × Mistral AI

<div style="display: flex; align-items: center; gap: 20px;">

  <div style="flex: 1;">

- Development of an in-vehicle voice assistant with conversational support and natural voice interaction, evolving through OTA updates.
- [Stellantis Press Release, 2025](https://www.stellantis.com/en/news/press-releases/2025/february/stellantis-and-mistral-ai-strengthen-strategic-partnership-to-enhance-customer-experience-vehicle-development-and-manufacturing)

  </div>

  <div style="flex: 1; text-align: center;">
    <img src="https://www.stellantis.com/content/dam/stellantis-corporate/news/press-releases/2025/february/07-02-25/Stellantis-Mistral.png" alt="Stellantis and Mistral AI Partnership" style="width: 100%; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    
  </div>

</div>

---

# <div style="display: flex; justify-content: center; align-items: center; height: 100vh; text-align: center; font-size: 10vw; font-weight: bold; width: 100%;">Legacy Code (e.g., PBC SW)</div>

---

# Challenges in Legacy Code

- **Sparse Documentation:** Outdated or missing documentation complicates understanding.
- **Complex Dependencies:** Intricate dependencies and fragile builds hinder updates.
- **High-Risk Modifications:** Altering core systems like Power Brake Control (PBC) software poses significant risks.

---

# Concrete Examples of AI on Legacy Code (Automotive & Embedded Systems)

| Use Case | Company / Authors | Description | Source |
|----------|-------------------|-------------|--------|
| **Understanding & Documenting Legacy Code** | Thoughtworks – CodeConcise | Use of LLMs to summarize and explain legacy C++ code, accelerating onboarding for engineers. | [Thoughtworks – CodeConcise](https://www.thoughtworks.com/codeconcise) |
| **Automated Migration (REXX → Java)** | IBM + AWS | Conversion of 100,000+ lines of REXX (Restructured Extended Executor) to Java in two weeks via an LLM pipeline, including documentation and diagram generation. | [IBM – Accelerating code modernization with generative AI (Automotive)](https://www.ibm.com/products/blog/accelerating-code-modernization-gen-ai-automotive) |
| **AI-Assisted Test-Driven Development** | Sibros (via Cursor/Claude Code) | Automatic generation of unit tests for critical functions (e.g., brake controllers), followed by LLM-guided refactoring. | [Thoughtworks – Claude Code experiment](https://www.thoughtworks.com/insights/blog/generative-ai/claude-code-codeconcise-experiment) |
| **Legacy Code Documentation & Review** | MITRE (Macke, Doyle) | Empirical study showing that GPT-4 produces complete and useful comments on old code (MUMPS, assembly). | [arXiv:2404.03114](https://arxiv.org/abs/2404.03114) |
| **Embedded Automotive Code Generation** | Sven Kirchner, Alois C. Knoll et al. | Framework for generating automotive C++ code using LLMs, with verification and simulation for safety-critical systems. | [arXiv:2506.04038](https://arxiv.org/abs/2506.04038) |
| **Spec2Code (Automotive Scania)** | Scania + Researchers | Generation of embedded code from specifications, with formal validation and critique tools. | [PapersWithCode – Spec2Code](https://paperswithcode.com/paper/towards-specification-driven-llm-based) |
| **Analysis of Hallucinations in Automotive Code** | Pavel, Petrovic et al. | Study of hallucinations in automotive code generation by LLMs, with metrics for syntactic and functional validity. | [arXiv:2508.11257](https://arxiv.org/abs/2508.11257) |

---

# [Thoughtworks – CodeConcise](https://www.thoughtworks.com/codeconcise) in details

Generative AI can help developers understand legacy codebases, especially when the documentation is poor, outdated, or misleading. 

Thoughtworks has developed a generative AI tool called CodeConcise, designed to help teams modernize legacy code. 

CodeConcise uses a Large Language Model (LLM) and a knowledge graph derived from Abstract Syntax Trees (ASTs) to analyze and document legacy code. 

AI can generate documentation more quickly and ensure it is geared toward the specific needs of business analysts. 

A particularly promising technique is a retrieval-augmented generation (RAG) approach where the information retrieval is done on a knowledge graph of the codebase.

---

# AI Contributions

## Code Retrieval
- Quickly locate relevant modules, functions, or APIs.
- **Example:** AI tools like **CodeWhisperer** and **GitHub Copilot** help developers navigate large legacy codebases efficiently.

## Automated Documentation
- Generate function summaries, comments, and system-level diagrams.
- **Example:** **Doxygen** combined with AI can automate documentation generation.

## Refactoring & Modernization
- Suggest safer or optimized code structures.
- Update outdated patterns.
- **Example:** AI-driven refactoring tools like **SonarQube** and **DeepCode** identify vulnerabilities and suggest improvements.

## Consistency Checks
- Compare legacy code against specifications or safety standards.
- **Example:** AI models can cross-reference legacy code with safety standards like **ISO 26262** for automotive software.

---

# AI for Legacy Code - Govtech Lab - Luxembourg

- **Legacy Java Application:**
  - ~700k lines of code.
  - Java 8, WebSphere 9, Struts/JSP/Vue.js.
  - Incomplete or outdated documentation.
  - Complex dependencies and control flows.
  - High effort for debugging or adding features.
  - Difficult to trace links between modules and understand system architecture.

---

# AI-Powered Solution for Legacy Code
  - Syntactic & semantic parsing of the codebase.
  - Function, class, and module descriptions.
  - Control flow and algorithm explanations.
  - Dependency mapping and UML diagrams (class, sequence, component).
  - Annotated source code and contextual glossary.

**Key Value**
- **Faster Understanding:** Of complex legacy systems.
- **Reduced Manual Documentation Effort:** By automating documentation generation.
- [Govtech Lab - Luxembourg](https://govtechlab.public.lu/fr/call-solution/2024/speedup-ailegacycode.html)

---

# Thermal Monitoring & Fading - Predictive Analysis with AI

<div style="display: flex; align-items: center; gap: 20px;">

  <div style="flex: 1;">

**Key message**
  - Brake fade is history-dependent. Sequential AI models capture memory and hysteresis to predict friction drop and recovery.

**Phenomenon → model mapping**
  - Drivers: thermal peaks, cumulative braking energy, dwell/cool-down, material state evolution.
  - Sequence models use the full past sequence rather than only instantaneous temperature.

  </div>
  <div style="flex: 1;">

  <img src="https://www.thermoanalytics.com/sites/default/files/styles/site_max/public/2020-02/thermal-simulation-of-brake-disc-and-caliper-showing-heat-generation-from-friction.png" alt="Thermal Simulation of Brake Disc and Caliper" style="width: 50%;">
  <p style="font-size: 0.9em; color: #555;">Thermal simulation of brake disc and caliper showing heat generation from friction.</p>

  </div>
</div>

---

## Thermal Monitoring & Fading - Predictive Analysis with RNN or Attention Mechanisms

**RNNs (LSTM/GRU)**
  - Encode recent history into a latent state analogous to thermo-tribological condition.
  - Good for short-to-medium memory; may struggle with very long sequences.

**Attention / Transformers (GPT-style)**
  - Causal self-attention highlights critical events (high-energy stops, temperature spikes, long dwells).
  - Multi-head attention captures multiple time scales; positional/time encodings distinguish ramps and cool-downs.

---

# Thermal Monitoring & Fading - Predictive Analysis with AI

**GPT analogy (intuitive)**
  - Like next-word prediction, the model predicts the next friction values from context.
  - Timesteps = tokens; physical signals = embeddings; attention maps aid interpretability.

**Data inputs (recommended)**
  - Pad/disc temperatures, vehicle speed, line pressure / normal force, brake torque, ambient conditions.
  - Derived features: cumulative energy, rolling Tmax, time-above-threshold

---

# Role of Attention Mechanism
- **Focuses on critical cycles** that most influence friction loss  
- Allows the model to **reuse important information from long sequences**  
- **Enhances interpretability**:  
  - Identifies which cycles, temperatures, pressures, or speeds drive fading  
  - Useful for **R&D validation and brake design**  
- Improves **prediction accuracy** when fading depends on **distant past cycles**  
- Considerations:  
  - Adds **computational complexity** → check real-time feasibility  
  - Most beneficial for **long sequences with cumulative effects**  

---

# 🔍 Attention Mechanism in Predictive Maintenance & Thermal Monitoring
| Article | Domain | Role of Attention | Source |
|---|--------|--------------------|------|
| *CNN-LSTM based on attention mechanism for brake pad remaining life prediction* | Brake pad life | Highlights relevant past time steps. | [Wang et al.](https://journals.viserdata.com/index.php/mes/article/view/9085) |
| *A deep attention based approach for predictive maintenance applications in IoT scenarios* | IoT maintenance | Multi-head attention for RUL estimation. | [De Luca et al.](https://www.emerald.com/insight/content/doi/10.1108/jmtm-02-2022-0093/full/html) |
| *Enhanced Thermal Modeling of Electric Vehicle Motors Using a Multihead Attention Mechanism* | EV motors | Captures complex thermal dynamics. | [MDPI Energies](https://doi.org/10.3390/en17122976) |
| *Thermal fault prognosis of lithium-ion batteries in real-world electric vehicles using self-attention mechanism networks* | EV batteries | Predicts thermal anomalies. | [Applied Thermal Eng.](https://doi.org/10.1016/j.applthermaleng.2023.120304) |
| *A Two-Stage Attention-Based Hierarchical Transformer for Turbofan Engine Remaining Useful Life Prediction* | Turbofan engines | Temporal and sensor-wise attention. | [MDPI Sensors](https://www.mdpi.com/1424-8220/24/3/824) |

---

<div style="display: flex; align-items: center; gap: 20px;">

  <div style="flex: 1;">

## May be Attention is all need for predicting Thermal Monitoring & Fading 
  
  </div>

  <div style="flex: 1;">
    <img src="https://image.slidesharecdn.com/attentionisallyouneed-180911075353/95/attention-is-all-you-need-1-638.jpg" alt="Attention is All You Need" style="width: 100%;">
  </div>

</div>


---

# AI-Driven Friction Material Characterization & Selection

---

# Material Characterization & Prediction with AI

[AI-Powered Prediction of Friction and Wear in Functionalized Epoxy-MWCNT Composites - ScienceDirect 2025](https://www.sciencedirect.com/science/article/pii/S0043164825006027)


<div style="display: flex; align-items: center; gap: 20px;">

  <div style="flex: 1;">

**Technologies:**
- **Artificial Intelligence** (Neural Networks, Generative Models, Autoencoders)
  - Advanced artificial intelligence methods were employed to predict key tribological parameters coefficient of friction (COF) and wear rate as well as to classify wear mechanisms

- **AI** models predict friction and wear in functionalised epoxy-MWCNT composites.

- **ANN** achieves high accuracy predicting coefficient of friction and wear rate.

- **RNN-LSTM** models capture time-dependent friction under varied wear mechanisms.

- **CNN** classifies abrasive, adhesive, fatigue, and delamination wear.

  </div>
  <div style="flex: 1;">

**Data Sources:**
- Tribometer logs (friction and wear data)
- Scanning Electron Microscope images (Hitachi SU-70) for (microstructural analysis)
- Chemical composition
  </div>
</div>
---

# Brake Composites Optimization with AI

<div style="display: flex; align-items: center; gap: 20px;">

  <div style="flex: 1;">

- Neural networks for predicting friction material performance.
- Generative AI (foundation models) for compound recipe design and iteration.
- Neural network-based computer vision for brake pad surface classification.

  </div>

  <div style="flex: 1;">
    <img src="https://media.geeksforgeeks.org/wp-content/uploads/20231207095717/logmel.png" alt="Brake Pad Surface Classification" style="width: 66%; border: 1px solid #ccc; border-radius: 8px;">
  </div>

</div>
---

# Brake Composites Optimization with AI

**Results (from industry cases):**

| Study / Case | Company | Description | Key Results | Source |
|--------------|---------|-------------|-------------|--------|
| *Revolutionising R&D: TMD Friction’s AI-driven path to the perfect friction formula* (2025) | **TMD Friction** | Uses in-house neural networks to virtually test and optimize new brake-friction material formulas. | Reduced dyno testing costs and time; AI models accurately predict ingredient effect on performance. | [Presse Release, TMD friction June 27th, 2025](https://tmdfriction.com/wp-content/uploads/2025/06/Press_release_AI_at-TMD_Friction.pdf) |
| *ALCHEMIX – Generative AI for Brake Compounds* (2024) | **Brembo** | Uses Microsoft Azure OpenAI to generate and evaluate novel brake pad recipes virtually. | Cut compound development time from days to minutes; AI explores unconventional formulations and flags errors. | [Brembo Press Release, 2024](https://brem-p-001.sitecorecontenthub.cloud/api/public/content/34df2592fb7542809cc94ce974b62e2b?v=c06d176b) |
| *AI Vision for Brake Pad Quality Inspection* (2020s) | **E.P.F. Elettrotecnica** | Siemens Simatic NPU runs a neural network to classify pad surface structures during production. | 80% less manual inspection; 100% pads automatically checked with higher consistency. | [Siemens Case Study](https://references.siemens.com/en/reference?id=23301) |

---

# Vision-based classification system for brake pads

<div style="display: flex; align-items: center; gap: 20px;">

  <div style="flex: 1;">

Reduced manual effort by 80% in classification process. 

Easy integration in existing automation system based on SIMATIC S7-1500. 

Automated inspection system allows 100% test coverage of brake pads.

  </div>

  <div style="flex: 1; text-align: center;">

<iframe width="560" height="315" src="https://www.youtube.com/embed/-FEVyaKAOew?si=ortamkPyqkdXX_py" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

  </div>
</div>


---

# Brake Noise Analysis with CNNs

**Technologies:**
- Faster R-CNN (Inception V2 backbone) applied on spectrogram images from brake acoustic test benches.  
- Standard CNN (e.g., ResNet) on audio spectrograms for squeal vs. moan classification.  

**Results (from industry cases):**

| Study / Case | Description | Key Results | Source |
|--------------|-------------|-------------|--------|
| *Brake NVH Noise Detection using Faster R-CNN* (Applus IDIADA, 2021) | Detection of squeal events on brake dyno acoustic data| Achieved ~**0.55 mAP** for squeal event detection. | [The BRAKE Report – IDIADA](https://thebrakereport.com/ai-as-a-means-of-detecting-brake-noise-part-2-of-3/) |
| *AI-Assisted NVH Classification* (Renumics, 2022) | CNN (ResNet-style) on spectrograms to distinguish squeal vs. moan, reducing false positives in durability and NVH tests. | Demonstrated higher classification accuracy vs. rule-based triggers (no % published). | [Renumics Blog – NVH Analysis](https://renumics.com/blog/ai-nvh-analysis/) |

---

# Generative / Optimization AI for Brake Composites — Industry

| Technology (AI, non-ML) | Industry Case / Company | Description & Role | Key Results / Notes | Source (URL) |
|-------------------------|-------------------------|--------------------|---------------------|--------------|
| **Generative Design / Topology Optimization** | **General Motors (Autodesk / Generative Design)** | GM used Autodesk generative design (cloud-based AI/optimization algorithms) to explore thousands of design alternatives (e.g., seat-bracket). Approach widely used in automotive for parts re-engineering (inspiration for lighter, manufacturable geometries). *Applicable to brake hardware design (calipers, carriers, discs) — not friction composite chemistry.* | Demonstrated weight reductions and part consolidation in GM prototypes (proof-of-concept). Generative design is production-ready for some OEM cases. | https://www.autodesk.com/customer-stories/general-motors-generative-design  |
| **Topology Optimization (structural / thermo-mechanical)** | **University → OEM collaborations (caliper / disc studies)** | Topology optimisation used to re-design calipers, discs, and pedals for weight and thermal performance; adopted as an industrial design toolchain (CAD → TO → AM / conventional manufacture). | Several industrial case studies & papers show improved thermal dissipation, lower mass and validated prototypes (used as design input in OEM labs). | https://www.mdpi.com/2076-3417/11/4/1437 (Topology optimization of brake calipers)  |
| **Genetic Algorithms / Multi-Objective Evolutionary Algorithms (NSGA-II)** | **Automotive R&D & SAE studies** | GA / NSGA-II applied to multi-objective parameter search (brake disc/pad geometry, process parameters) to trade off friction, wear and weight; typically combined with physics simulations (FEM / tribology models). | Papers and SAE tech-papers demonstrate resulting Pareto fronts and optimized parameter sets (reduced mass, maintained performance). | SAE technical paper: Optimization of Brake System Parameters Using Genetic ... (2023) https://www.sae.org/publications/technical-papers/content/2023-01-1881/  |
| **Design of Experiments (DOE) / Taguchi for materials/process optimisation** | **Industrial friction labs / suppliers (various)** | DOE and Taguchi methods remain standard in R&D for formulation tuning (curing time, pressure, filler ratios). They are often used alongside AI optimisation loops to seed/evaluate candidate formulations. | Industry reports use DOE to reduce test matrix size and identify robust process windows (widely referenced in friction materials literature). | Example: Optimization of Process Parameters for Friction Materials (MDPI, 2021) https://www.mdpi.com/2227-9717/9/9/1570  |
| **Physics-based surrogate optimisation & virtual testing (non-ML solvers)** | **Tier-1 R&D workflows (conceptual / vendor case studies)** | Use physics solvers (FEM thermal/tribological models) coupled to optimisation engines (gradient/free-form optimizers, GA) to evaluate candidates virtually before dyno tests. | Reported industrial benefit: fewer dyno cycles and faster iteration (case studies from supplier whitepapers). Quantitative gains depend on setup. | Example overview: Generative Design & Topology Optimization (Autodesk) https://www.autodesk.com/design-make/articles/generative-design-in-manufacturing  |

<!--**Notes & limitations:**
- There are *many* academic papers applying GA / NSGA-II / TO to brake components (calipers, discs, pedals) and process parameters; industrial deployment is often proof-of-concept or R&D-centered rather than mass production for friction **formulation** (pad chemistry) specifically.  
- For **friction composite chemistry (pad formulations)**, most recent industrial advancements combine physics simulations + ML surrogates; true industry examples that use *only* non-ML AI for automatic compound generation are rare in public literature. Suppliers (TMD Friction, Brembo, Fras-le) report “AI & virtual testing” in R&D, but often do not disclose whether ML or optimisation algorithms are used internally. See Brembo ALCHEMIX (uses Azure OpenAI — ML) and TMD Friction press items (R&D AI) for industry traction (these are ML-centric and were intentionally **excluded** from the table above because you asked to exclude ML). | Brembo ALCHEMIX news: https://www.brembogroup.com/en/media/news/alchemix-microsoft (note: uses Azure OpenAI — ML)  |-->

---

# The Impact of AI on Automotive Testing, Simulation, and Performance

| Metric                     | Traditional Process                        | AI-Augmented Workflow                                      | Main Source(s)                                                                                                                                                                                                 |
|----------------------------|--------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| NVH Complaint Rate         | Higher complaint rates, late detection     | Up to 60% reduction through predictive analysis             | [Automotive Quality](https://www.automotivequal.com/artificial-intelligence-in-quality-management-case-studies/), [SAE International](https://www.sae.org/publications/technical-papers/content/2024-01-2927/)                                                      |
| Crash Simulation Accuracy  | Physics-driven solvers, time-consuming     | Reduction in simulation time by 3 orders of magnitude, >98% accuracy | [CORDIS/EU - UPSCALE Project](https://cordis.europa.eu/article/id/444119-advanced-algorithms-improve-electric-vehicle-efficiency/fr)                                                                                     |

---
# AI as a Lever for Innovation in the Automotive Industry

**Key Opportunities:**
- **Process Optimization:** Cost and time reduction (e.g., 50% faster prototyping).
- **Safety Enhancement:** Predictive maintenance, traffic sign recognition, and voice-assisted driving.
- **User Experience:** Personalized in-car interactions and comfort (e.g., advanced voice assistants).
- **Sustainability:** Material optimization and waste reduction through AI-driven design.

---
# Challenges and Future Outlook

**Current Challenges:**
- Data quality and system interoperability.
- User acceptance and ethical considerations (e.g., algorithm transparency).
- Integration with existing production lines.

**Future Perspectives:**
- Autonomous and connected vehicles.
- Collaboration between automakers and tech startups.
- Regulatory frameworks and standardization.

---
# Call to Action: Launching Your AI Project in Automotive

**Where to Start?**
1. **Define a Clear Problem:**
   - Focus on a specific pain point (e.g., reducing brake pad wear, improving voice recognition accuracy).
   - Ensure the problem is measurable and aligned with business goals.

2. **Secure High-Quality Data:**
   - Collect relevant, clean, and labeled data in sufficient volume.
   - Leverage existing datasets or invest in data collection/annotation.

3. **Build or Partner:**
   - Develop in-house expertise or collaborate with AI specialists.
   - Use open-source tools or cloud platforms (e.g., AWS SageMaker, Google Vertex AI).

**Pro Tip:**
*"Success in AI starts with a well-defined problem and high-quality data—scale comes later."*

---

# 🛠️ Practical Workshops

## **Part 2 – Interactive Workshops (1h30)**

**Access the workshops:**
📋 **[AI Workshops Interface](workshops_standalone.html)**

---

**Available workshops:**

🔍 RAG and Technical Documentation

💻 Legacy Code: Documentation & Maintenance  

⚙️ AI-assisted Mechanical Design

📊 Software Specifications & Deviation Matrices

🌡️ Virtual Sensors & Indirect Estimation

🚗 Thermal Monitoring & Vehicle Dynamics

🧪 Testing Optimization & Simulation Enhancement

**Format:** Small groups (3-5 participants) • Objective-driven • Practical solutions

📋 **[AI Workshops Interface](workshops_standalone.html)**

---

# Q&A: Your Turn to Drive the Conversation

**Let’s discuss:**
- What AI challenges are you facing in your projects?
- How do you see AI transforming your role in the automotive industry?
- Any specific use cases you’d like to explore further?

*"No question is too small—let’s make AI work for you!"*
---

# <div style="display: flex; justify-content: center; align-items: center; height: 100vh; text-align: center; font-size: 10vw; font-weight: bold; width: 100%;">Thank You!</div>

**Contact:**

## [💼 LinkedIn Famibelle](https://www.linkedin.com/in/famibelle/)

---

## 💻 Available Resources for the Seminar


<div style="display: flex; align-items: center; justify-content: space-around;">

<div>

🔗 [GitHub Repository](https://github.com/famibelle/AISeminar)


📝 [Slides](https://github.com/famibelle/AISeminar/blob/1db270e2e298202205c81305eb3ee3bdfbccdb44/ai_seminar_slides.md)


📖 [AI Labs](https://github.com/famibelle/AISeminar/tree/1db270e2e298202205c81305eb3ee3bdfbccdb44/Labs)  


🛠️ [YOLO Tools](https://github.com/famibelle/AISeminar/blob/1db270e2e298202205c81305eb3ee3bdfbccdb44/Labs/yolo_reveal_auto.py)

</div>

<div>

![GitHub Repository QR Code](https://api.qrserver.com/v1/create-qr-code/?size=280x280&data=https://github.com/famibelle/AISeminar/)

**📱 Scan to access**

</div>

</div>