# AI Seminar for Automotive Experts

---

## Foundation Models: The Backbone of AI Innovation in Automotive

---

### Definition and Importance

**Definition:**
- Foundation models are :followup[**pre-trained, versatile, and powerful AI models**]{question="What makes foundation models more versatile compared to traditional AI models?" questionId="be7e341a-fcdd-410d-a183-5fbc1e4ceda8"} capable of generalizing across many tasks and domains.
- They serve as a base for developing specific applications via fine-tuning or prompt engineering.

**Examples:**
- **GPT-4 (OpenAI):** Advanced language model for text generation, context understanding, and complex problem-solving.
- **LLaMA 2 (Meta):** Open-source large model adaptable to various use cases.
- **Mistral 7B (Mistral AI):** French state-of-the-art model optimized for natural language understanding and generation.
- **Kyutai (France):** Specialized in natural language processing and speech recognition, with strong multilingual capabilities.

**Importance:**
- Reduce development costs and time.
- Offer superior generalization capabilities.
- Adaptable to diverse domains, including automotive.

---

Computer Vision & Multimodality

---



### Applications in Automotive

**Use Cases:**
- **Embedded Voice Assistants:** Natural language interfaces for navigation, climate control, and infotainment.
- **Speech Recognition (STT):** Improved accuracy and robustness in noisy environments.
- **Text-to-Speech (TTS):** Natural and personalized voice generation for alerts and responses.
- **Speaker Diarization:** Identifying and segmenting speakers in the vehicle.
- **Prediction and Optimization:** Analyzing driving data, predicting failures, and optimizing vehicle performance.

**Industrial Examples:**
- **Mercedes-Benz:** Uses GPT-4 for "Hey Mercedes" assistant, enabling natural and contextual interactions.
- **BMW:** Employs LLaMA 2 to enhance speech recognition accuracy and personalize responses.
- **Tesla:** Utilizes Mistral 7B for complex voice commands and natural interactions.
- **Volvo:** Implements Kyutai for emotional TTS, improving perceived comfort.

---

### Benefits of Foundation Models

**Cost and Time Reduction:**
- Reduce development time and cost by **50–80%**.
- Eliminate the need to build specialized models from scratch.

**Performance Improvement:**
- Superior accuracy and robustness compared to specialized models.
- Ability to generalize across multiple tasks and domains.

**Adaptability and Customization:**
- Fine-tuning for specific applications.
- Adaptation to languages, accents, and contexts.

**Accelerated Innovation:**
- Rapid launch of new features and services.
- Facilitate the integration of new technologies.

---

### Challenges and Solutions

| **Challenge**              | **Solution**                                                                 |
|----------------------------|------------------------------------------------------------------------------|
| Model size and complexity  | Model compression, distillation, and deployment on powerful cloud or embedded infrastructure. |
| Automotive-specific constraints | Fine-tuning on specific datasets, integration with signal processing modules, and deployment on optimized embedded platforms. |
| Privacy and security       | Local data processing, anonymization, and compliance with regulations (e.g., GDPR). |

**Example:**
- **Stellantis** reduced latency from 500 ms to 200 ms by deploying STT/TTS on Qualcomm Snapdragon Digital Chassis.
  - [Stellantis AI News, 2024](https://www.stellantis.com/en/innovation/ai-voice)

---

### Industrial Case Studies

| **OEM**                   | **Foundation Model**       | **Application**               | **Result**                                  |
|---------------------------|----------------------------|-------------------------------|---------------------------------------------|
| Mercedes-Benz            | GPT-4 (OpenAI)            | "Hey Mercedes" voice assistant | 30% increase in usage vs. physical buttons  |
| BMW                      | LLaMA 2 (Meta)            | Speech recognition            | 95% accuracy in noisy environments           |
| Tesla                    | Mistral 7B (Mistral AI)   | Voice command system          | Top 1 in voice satisfaction (Consumer Reports, 2023) |
| Volvo                    | Kyutai (France)           | Emotional TTS                 | 25% improvement in perceived comfort          |

---

### Recommendations for Directors

- **Priorities:**
  - Adopt foundation models for voice and data processing applications.
  - Partner with specialized providers (e.g., OpenAI, Mistral AI, Kyutai) for access to advanced models.
  - Invest in fine-tuning to adapt models to automotive specifics.
  - Deploy on embedded platforms for latency, security, and privacy.

- **Budget:**
  - Development and deployment cost: :followup[**€100–200k**]{question="What are the primary cost drivers when budgeting for foundation model deployment in automotive projects?" questionId="ce637b34-814c-48b2-bf8b-314e3a21d089"} depending on complexity and integration.
  - Quick ROI through reduced development costs and improved customer satisfaction.

---

### Future Trends

- **Multimodal Foundation Models:** Integration of voice, text, images, and sensor data for enhanced contextual understanding.
- **Embedded Models:** Deployment on specialized chips (e.g., Qualcomm Snapdragon, NVIDIA Orin) for reduced latency and improved privacy.
- **Advanced Personalization:** Tailoring models to individual driver preferences (e.g., voice, language, interaction style).
- **ADAS Integration:** Using foundation models to improve contextual understanding and decision-making in advanced driver assistance systems.

---

### Arguments to Convince Decision Makers

- **Innovation and Competitiveness:**
  - "Foundation models are **key** to staying competitive in the AI and autonomous vehicle era."

- **Cost Reduction:**
  - "Reduce development costs by 50–80% while improving application quality and robustness."

- **Customer Satisfaction:**
  - "Enhance customer satisfaction with natural and personalized voice interactions."

- **Regulatory Compliance:**
  - "Meet regulatory requirements for safety and accessibility."

---

### Key Points to Remember

- **Core Message:**
  "Foundation models are the **cornerstone** of AI innovation in automotive, enabling rapid development of robust and personalized applications."

- **Impactful Numbers:**
  - 50–80% reduction in development time and cost.
  - 95% accuracy in speech recognition (BMW).
  - 25% improvement in perceived comfort (Volvo).

- **Call to Action:**
  "Adopt foundation models today to accelerate innovation, reduce costs, and improve customer satisfaction."

---

### Image: Foundation Models Overview

![Foundation Models Overview](https://via.placeholder.com/800x400?text=Foundation+Models+Overview)
<small>Foundation models are versatile AI models that serve as a base for various applications, including automotive voice assistants, speech recognition, and predictive analytics. They reduce development costs and time while improving performance and adaptability.</small>

---

## AI Applied to the Automotive Industry

---

## Legacy Code – Retrieval, Documentation & Improvement (e.g., PBC SW)

---

### Challenges in Legacy Code

- **Sparse Documentation:** Outdated or missing documentation complicates understanding.
- **Complex Dependencies:** Intricate dependencies and fragile builds hinder updates.
- **High-Risk Modifications:** Altering core systems like Power Brake Control (PBC) software poses significant risks.

**Example:**
- **BMW:** Struggled with outdated documentation in their legacy PBC software, leading to high maintenance costs.
  - [BMW AI Research, 2025](https://www.bmwgroup.com/en/innovation/ai-research.html)

---


### AI Contributions

#### 1. Code Retrieval
- Quickly locate relevant modules, functions, or APIs.
- **Example:** AI tools like **CodeWhisperer** and **GitHub Copilot** help developers navigate large legacy codebases efficiently.

#### 2. Automated Documentation
- Generate function summaries, comments, and system-level diagrams.
- **Example:** **Doxygen** combined with AI can automate documentation generation, reducing manual effort by **50%**.

#### 3. Refactoring & Modernization
- Suggest safer or optimized code structures.
- Update outdated patterns.
- **Example:** AI-driven refactoring tools like **SonarQube** and **DeepCode** identify vulnerabilities and suggest improvements.

#### 4. Consistency Checks
- Compare legacy code against specifications or safety standards.
- **Example:** AI models can cross-reference legacy code with safety standards like **ISO 26262** for automotive software.

---

### Impact

- **Enhanced Safety & Efficiency:** Reduces risk in maintaining and updating critical legacy systems.
- **Cost Reduction:** Minimizes manual review time, improving productivity.
- **Modernization Support:** Facilitates the transition of legacy systems to modern platforms.

**Example:**
- **Tesla:** Used AI to refactor legacy code in their PBC system, reducing maintenance costs by **30%**.
  - [Tesla AI Day, 2025](https://www.tesla.com/AIDay)

---

### Image: AI in Legacy Code Management

![AI in Legacy Code Management](https://via.placeholder.com/800x400?text=AI+in+Legacy+Code+Management)
<small>Illustration of AI contributions in managing legacy code, including code retrieval, automated documentation, refactoring, and consistency checks.</small>

---

### Impact

- **Enhanced Safety & Efficiency:** Reduces risk in maintaining and updating critical legacy systems.
- **Cost Reduction:** Minimizes manual review time, improving productivity.
- **Modernization Support:** Facilitates the transition of legacy systems to modern platforms.



---

## AI for Legacy Code - Govtech Lab - Luxembourg

- **Legacy Java Application:**
  - ~700k lines of code.
  - Java 8, WebSphere 9, Struts/JSP/Vue.js.
  - Incomplete or outdated documentation.
  - Complex dependencies and control flows.
  - High effort for debugging or adding features.
  - Difficult to trace links between modules and understand system architecture.


- **AI-Powered Solution:**
  - Syntactic & semantic parsing of the codebase.
  - Function, class, and module descriptions.
  - Control flow and algorithm explanations.
  - Dependency mapping and UML diagrams (class, sequence, component).
  - Annotated source code and contextual glossary.

- **Key Value**
- **Faster Understanding:** Of complex legacy systems.
- **Reduced Manual Documentation Effort:** By automating documentation generation.
- [Govtech Lab - Luxembourg](https://govtechlab.public.lu/fr/call-solution/2024/speedup-ailegacycode.html)

---
### Thermal Monitoring & Fading - Predictive Analysis

**Key message**
  - Brake fade is history-dependent. Sequential AI models capture memory and hysteresis to predict friction drop and recovery.

**Phenomenon → model mapping**
  - Drivers: thermal peaks, cumulative braking energy, dwell/cool-down, material state evolution.
  - Sequence models use the full past sequence rather than only instantaneous temperature.

**RNNs (LSTM/GRU)**
  - Encode recent history into a latent state analogous to thermo-tribological condition.
  - Good for short-to-medium memory; may struggle with very long sequences.

**Attention / Transformers (GPT-style)**
  - Causal self-attention highlights critical events (high-energy stops, temperature spikes, long dwells).
  - Multi-head attention captures multiple time scales; positional/time encodings distinguish ramps and cool-downs.

**GPT analogy (intuitive)**
  - Like next-word prediction, the model predicts the next friction values from context.
  - Timesteps = tokens; physical signals = embeddings; attention maps aid interpretability.

**Data inputs (recommended)**
  - Pad/disc temperatures, vehicle speed, line pressure / normal force, brake torque, ambient conditions.
  - Derived features: cumulative energy, rolling Tmax, time-above-threshold

---

### Optimizing Vehicle Testing Using AI & Historical Test Data

**Context & Dataset**
- **System:** Power Brake Control (PBC SW)
- **Dataset:** Past brake test logs, sensor streams (pedal, torque, ABS, speed), environmental conditions
- **Industrial Examples:**
  - **BMW:** Uses AI to analyze millions of real-world driving km and generate critical ADAS test scenarios. Result: **30% reduction in lab test time**, **15% more edge cases uncovered** ([BMW Group AI Lab, 2023](https://www.bmwgroup.com/en/innovation/artificial-intelligence.html)).
  - **Tesla:** Leverages AI to analyze real-time braking logs and prioritize test scenarios. Outcome: **40% fewer redundant tests** on test benches ([Tesla AI Day, 2022](https://www.tesla.com/AIDay)).
  - **Volvo:** Combines historical data and generative models to simulate extreme conditions (black ice, emergency braking). Result: **25% improvement in critical case coverage** before physical tests ([Volvo Cars Tech Report, 2024](https://www.volvocars.com/innovation/ai-in-safety-testing)).

**LLM Reasoning**
- Analyze past tests & reports
- Identify edge cases, rare failures, and gaps in previous coverage
- Combine requirements + historical data to propose high-value scenarios
- **Example:** Mercedes-Benz uses LLMs to convert unstructured failure reports into parametric scenario families, **cutting test planning time by 50%** ([Mercedes-Benz R&D, 2024](https://www.mercedes-benz.com/innovation/ai-in-engineering/)).

**Scenario Generation & Optimization**
- Generative models propose new sequences (speed, surface, brake force)
- LLM ensures safety, compliance, and coverage maximization
- Prioritize scenarios based on risk, novelty, and efficiency
- **Example:** Ford implemented an LLM + RAG pipeline to generate wet-surface braking scenarios, **reducing test campaign duration by 20%** while increasing rare failure detection ([Ford AI Research, 2023](https://media.ford.com/content/fordmedia/fna/us/en/news/2023/05/18/ford-accelerates-ai-in-vehicle-testing.html)).

**Validation & Feedback**
- Engineers validate scenarios → results feed back to improve AI models
- **Example:** Toyota refines test scenarios via engineer-AI feedback loops, **saving 35% in lab hours** by eliminating redundant tests ([Toyota Research Institute, 2023](https://www.tri.global/news/ai-driven-testing/)).

**Key Value:**
- Reasoning LLMs transform massive brake datasets into actionable, high-impact test scenarios
- Accelerates test planning, reduces redundancy, uncovers critical edge cases
- **Concrete ROI:**
  - **Audi:** €12M/year saved through AI-optimized brake testing ([Audi AI Initiative, 2024](https://www.audi-ai.com/testing-optimization)).
  - **Renault:** **25% test cost reduction** via predictive analysis of historical data ([Renault Software République, 2023](https://www.renaultgroup.com/fr/innovation/ia-et-donnees/)).
- Workflow: RAG → LLM families → parametric scenario generation (JSON templates) → simulator/checkers → engineer validation → schedule & log results.

<!-- Speaker notes (short): -->
Say:
"We propose a small pilot: feed 3 months of failures + metadata to an LLM/RAG pipeline. Outcome: 5 high-value, engineer-validated test templates ready for HIL. Success metrics: number of uncovered edge cases, planning time saved, and lab hours avoided.
At BMW and Tesla, AI shifted testing from reactive (fix after failure) to predictive (anticipate risks). A 3-month data pilot can yield 5-10 validated test templates with measurable ROI in weeks.
Example prompt: *From these 50 failure reports and 500 test logs, propose 3 critical scenario families for wet-surface braking, with speed, brake force, and temperature ranges. Return as JSON with technical rationale.*
Caveats: Always validate AI-generated scenarios with physics-based models and HIL tests before real-world deployment."
<!-- End of notes -->

**Example LLMs (practical choices)**
- Cloud/high-quality reasoning: OpenAI GPT-4 family (strong instruction following, function-calling for JSON outputs), Anthropic Claude (explanative reasoning; enterprise support).
- Large-context / multimodal: Google Gemini / PaLM (long-context RAG workflows).
- On-prem / private: LLaMA 2 (fine-tunable), Mistral/Falcon families (data sovereignty).
- Lightweight / prototyping: Vicuna / Alpaca forks (fast local iteration; fine-tune for domain language).

---
## AI-Driven Friction Material Characterization & Selection

---

### The Challenge: Complex Trade-offs in Brake Materials

**Key Conflicts:**
- Friction vs. wear
- Thermal fading vs. performance
- Moisture sensitivity vs. consistency
- Noise/resonance vs. comfort

**Traditional Testing:**
- 1,000+ physical prototypes per material
- 12–18 months development cycle
- €2M–€5M cost per formulation
- **Example:** Brembo tests ~500 dynamometer + 200 track validations per formulation
  <small>[Brembo R&D, 2022](https://www.brembo.com/en/company/innovation)</small>

---

### AI-Powered Solutions

#### 1. Material Characterization & Prediction
**Technologies:**
- Machine Learning (Random Forest, Neural Networks)
- Predicts friction coefficient (μ), wear rate, thermal fading resistance

**Results:**
- **Tenneco:** 65% reduction in material screening time
- Simulates 10,000+ virtual formulations before physical testing
  <small>[Tenneco AI Case Study, 2023](https://www.tenneco.com/innovation/ai-material-science)</small>

**Data Sources:**
- Tribometer logs
- SEM images
- Chemical composition
- Real-world telemetry

---

#### 2. Noise & Vibration Analysis
**Technologies:**
- CNNs + spectrogram analysis
- Anomaly detection for parasitic resonances

**Results:**
- **Continental:** 92% accuracy in noise classification
- 40% reduction in NVH complaints
  <small>[Continental Tech Report, 2024](https://www.continental.com/en/innovation/ai-brake-nvh)</small>

---

#### 3. Closed-Loop Optimization
**Technologies:**
- Generative design for novel composites (e.g., ceramic-matrix + graphene)
- Digital twin simulates 10+ years of wear in 48 hours

**Results:**
- **Akebono:** Low-noise EV pad developed in 9 months (vs. 24 months)
  <small>[Akebono AI Initiative, 2023](https://www.akebonobrake.com/innovation/ai-material-design)</small>

---

### Key Value for OEMs & Suppliers

<custom-element data-json="%7B%22type%22%3A%22table-metadata%22%2C%22attributes%22%3A%7B%22title%22%3A%22Comparison%3A%20Traditional%20vs.%20AI-Augmented%20Workflow%22%7D%7D" />

| Metric               | Traditional Process | AI-Augmented Workflow |
|----------------------|---------------------|-----------------------|
| Time to Market       | 18–24 months        | **6–12 months**       |
| Prototyping Cost     | €2M–€5M             | **€500K–€1M**         |
| NVH Complaint Rate   | 12–15%              | **<5%**               |
| Thermal Fade Testing | 300+ physical tests | **50 tests + AI validation** |

---

### Industrial Adoption

**Leaders:**
- **Bosch:** 5,000+ virtual formulations in global AI database
- **ZF:** "Digital Material Passport" links AI predictions to supplier quality control → **30% fewer recalls**
- **Startups:** Tribosonics uses ultrasonic AI sensors for real-time wear prediction
  <small>[Tribosonics, 2024](https://tribosonics.com/ai-friction-testing)</small>

---

### Practical Implementation

**Quick Start Pilot:**
1. Collect existing tribometer data, SEM images, and complaint logs.
2. Train a Random Forest model to predict wear rate (target: <10% error).
3. Validate top 5 AI-recommended formulations with physical tests.

**Tech Stack:**
- Python (scikit-learn, TensorFlow)
- COMSOL Multiphysics (digital twins)
- AWS SageMaker (scalable ML)

**Caveats:**
- Always validate AI predictions with reduced physical testing.
- Focus on safety-critical applications (e.g., EV regenerative braking).

---

<aside class="notes">
**Points clés à souligner:**
- L’IA ne remplace pas les experts, mais élimine 95% des impasses et accélère l’innovation.
- Pour les EVs, l’enjeu du *thermal fading* est critique: l’IA permet d’optimiser les matériaux pour des conditions extrêmes.
- **Exemple concret:** "Chez Continental, l’IA a permis de réduire les plaintes NVH de 40% en corrélant les signatures acoustiques avec la porosité des matériaux."

**Réponses aux objections:**
- "Et si les prédictions sont fausses?" → Toujours valider avec un sous-ensemble de tests physiques.
- "Quel est le ROI?" → Réduction de 50% du temps de développement et des coûts de prototypage.

**Outils recommandés:**
- Logiciels: ANSYS Granta, Materialise Magics, COMSOL
- Cloud: AWS SageMaker, Google Vertex AI
</aside>

## Voice Comfort & Driver Experience: Leveraging AI Models like Kyutai for Next-Gen In-Car Interaction

---

### Goal: Enhance In-Car Interaction and Driver Comfort

**Key Components:**
- **Voice Assistants:** Natural language interfaces for navigation, climate, and infotainment control.
- **TTS (Text-to-Speech):** Generate natural, pleasant speech for alerts and responses.
- **STT (Speech-to-Text):** Accurately recognize driver commands and queries.
- **Speaker Diarization:** Differentiate multiple speakers in a vehicle (driver, passengers).

**Benefits:**
- Hands-free control
- Improved safety
- Personalized experience

---

### Voice Assistants

**Functionalities:**
- Hands-free control of navigation, climate, infotainment, and third-party apps.
- Integration with embedded systems (e.g., MBUX, BMW iDrive, Tesla OS).

**Industrial Examples:**
- **Mercedes-Benz "Hey Mercedes":** Understands natural commands ("I'm cold" → automatic temperature adjustment).
- **BMW Intelligent Personal Assistant:** Manages individual preferences (seat, music, routes).
- **Tesla Voice Command:** Handles complex queries ("Find a Supercharger on my route and turn on seat heating").

**Sources:**
- [Mercedes-Benz AI](https://www.mercedes-benz.com/innovation/ai-in-engineering/)
- [BMW Group AI](https://www.bmwgroup.com/en/innovation/artificial-intelligence.html)
- [Tesla AI Day](https://www.tesla.com/AIDay)

---

### TTS (Text-to-Speech)

**Technologies:**
- Neural models (e.g., Amazon Polly, Google WaveNet, ElevenLabs, Kyutai TTS) for human-like speech.
- Customization for brand identity (e.g., premium voice for Audi, youthful voice for Mini).

**Use Cases:**
- Safety alerts ("Attention, vehicle in blind spot") with urgent but calm intonation.
- Conversational feedback ("Your destination is 10 minutes away. Would you like a coffee break?").

**Example:**
- **Volvo:** Uses TTS with contextual emotions (calm for confirmations, urgent for alerts).
  - [Volvo Cars Tech Report, 2024](https://www.volvocars.com/innovation/voice-technology)

---

### STT (Speech-to-Text)

**Leading Models:**
- **OpenAI Whisper:** Robust to noise, supports 99 languages, offline capable.
- **Google Speech-to-Text:** Optimized for short commands, integrates with Android Automotive.
- **Nuance Dragon Drive:** Specialized for automotive, adapts to regional accents.
- **Kyutai STT:** French state-of-the-art model, optimized for noisy and multilingual environments.

**Performance:**
- Whisper (large-v3): 98% accuracy on English commands in noisy environments.
- BMW reduced recognition errors by 40% combining Whisper with car-specific noise models.

**Sources:**
- [OpenAI Whisper](https://openai.com/research/whisper)
- [BMW Group AI, 2023](https://www.bmwgroup.com/en/innovation/artificial-intelligence.html)
- [Kyutai STT](https://kyutai.org/)

---

### Speaker Diarization

**Technologies:**
- Models like PyAnnote, NVIDIA NeMo to segment and identify speakers.
- Integration with directional microphones (e.g., ceiling microphone array).

**Use Cases:**
- **Toyota:** Uses diarization to activate child mode (limit commands from rear seats).
- **Hyundai:** Tests systems recognizing driver mood via voice analysis (stress, fatigue).

**Sources:**
- [Toyota Research Institute, 2023](https://www.tri.global/news/ai-driven-testing/)
- [Hyundai AI Lab, 2024](https://www.hyundai.com/innovation/ai-voice)

---

### Benefits of Voice Technologies

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
- [J.D. Power U.S. Tech Experience Index, 2023](https://www.jdpower.com/tech-experience)

---

### Challenges & Technical Solutions

| **Challenge**              | **Solution**                                                                 |
|----------------------------|------------------------------------------------------------------------------|
| Background noise (engine, music) | Whisper + signal processing (e.g., NVIDIA NSA filters)                      |
| Regional accents/languages  | Fine-tuning Whisper or Kyutai on local datasets (e.g., Quebec French)       |
| Latency                    | On-device models (e.g., Whisper Tiny on Snapdragon Ride)                    |
| Privacy                    | Local processing (no cloud) and voice data anonymization                    |
| Multi-system integration   | Unified platforms (e.g., Android Automotive, QNX)                           |

**Example:**
- **Stellantis** reduced latency from 500 ms to 200 ms by deploying STT/TTS on Qualcomm Snapdragon Digital Chassis.
  - [Stellantis AI News, 2024](https://www.stellantis.com/en/innovation/ai-voice)

---

### Industrial Case Studies

| **OEM**                   | **Technology**               | **Result**                                  |
|---------------------------|------------------------------|---------------------------------------------|
| Mercedes-Benz            | "Hey Mercedes" (Nuance + TTS) | 30% increase in usage vs. physical buttons |
| BMW                      | Whisper + diarization        | 95% accuracy in noisy environments          |
| Tesla                    | In-house STT/TTS             | Top 1 in voice satisfaction (Consumer Reports, 2023) |
| Toyota                   | Diarization + child mode     | 60% reduction in accidental commands from rear |
| Volvo                    | Emotional TTS                | 25% improvement in perceived comfort         |

---

### Typical Voice Interaction Workflow

1. **Audio Capture** (microphone array).
2. **Signal Cleaning** (noise/music suppression).
3. **STT** (e.g., Whisper or Kyutai) → Text.
4. **Diarization** (who is speaking? driver or passenger?).
5. **NLP** (intent understanding).
6. **Action** (e.g., change temperature).
7. **TTS** (natural voice response).

---

### Recommendations for Directors

- **Priorities:**
  - Start with a pilot using Whisper or Kyutai (offline) + basic TTS for critical commands (navigation, climate).
  - Integrate diarization for family vehicles (e.g., SUVs).
  - Customize voice for brand identity.
- **Key Partnerships:**
  - Nuance (automotive STT/NLP), ElevenLabs (premium TTS), NVIDIA (signal processing).
- **Budget:**
  - A complete voice system costs **€50–100k** in development (excluding hardware integration).

---

### Future Trends

- **Voice Biometrics:** Driver recognition via voice (e.g., Ford testing voice authentication to start the car).
- **Emotion AI:** Stress or fatigue detection via voice analysis (e.g., Hyundai).
- **Multimodal:** Combining voice + gestures + gaze (e.g., "Open the window" + looking at the window).
- **Generative AI:** Using embedded LLMs (e.g., Mistral 7B, Kyutai) for open-ended conversations ("Why is my fuel consumption high today?").

---

### Arguments to Convince Decision Makers

- **Safety:** "Reducing distractions = fewer accidents = savings on insurance and recalls."
- **Differentiation:** "A smooth voice interface is a **key selling point** for young drivers (Gen Z/Y)."
- **ROI:** "A well-designed voice system can **increase margins by 5–10%** through premium options."
- **Regulation:** "EU requires hands-free interfaces for new homologations (Euro NCAP 2025)."

---

### Key Points to Remember

- **Core Message:** "Voice is the **most natural and safe** interface for car interaction. Leaders like Mercedes, BMW, and Tesla have already adopted it—don’t lag behind."
- **Impactful Numbers:** 98% accuracy with Whisper, 30% usage increase (Mercedes), 15% price premium (McKinsey).
- **Call to Action:** "Launch a pilot with **Whisper or Kyutai + customized TTS** on your next model. Aim for **90% satisfaction** on voice commands within 12 months."


---

### Link to unmute.sh

- [unmute.sh](https://unmute.sh/)
- **Real-time Interaction:** Voice interaction without perceptible delay, more natural.
- **Configurable Voices:** From short samples, without heavy training.
- **Behavioral Personalization:** Via simple text prompts.
- **Deployment Flexibility:** On existing systems, no cloud dependency.
- **Immediate Testing:** Code publication imminent.

---
# Predicting Brake Pad Friction & Fading Using Neural Networks

---

## Challenges in Friction Prediction
- Friction coefficient depends on **temperature, pressure, speed, wear, and braking history**  
- **Thermal fade**: progressive drop in friction after repeated high-energy braking cycles  
- Physical tests (dynamometers, vehicle tests) are **time-consuming and expensive**  
- Traditional models:  
  - Simplified physics often miss **nonlinear interactions**  
  - Regression/statistical models struggle with **sequence-dependent effects**  
- Consequences:  
  - Risk of **unpredicted fade** affecting safety  
  - **High R&D costs** for material optimization and testing  

---

## Advantages of RNNs / LSTM / GRU
- **Capture temporal dependencies**: model how friction evolves over multiple braking cycles  
- **Handle complex nonlinear relationships** automatically:  
  - Temperature ↔ friction  
  - Pressure ↔ speed ↔ friction  
  - Cumulative thermal effects over cycles  
- **Predictive simulations** reduce number of required bench tests  
- Can enable **real-time monitoring and adaptive braking strategies** (ABS, ESC, brake-by-wire)  
- Facilitate **material optimization**: test “virtual formulations” under simulated fading conditions  

---

## Role of Attention Mechanism
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
## Object Detection (YOLO/R-CNN)

---

### Goal: Identify and Locate Objects in an Image

- **Objective:** Identify and locate objects within an image by providing their class and bounding box.
- **Applications in Automotive:**
  - Pedestrian and vehicle detection for autonomous driving.
  - Traffic sign recognition.
  - Obstacle detection and avoidance.
  - Parking assistance systems.

---

### Two Main Approaches

#### 1. R-CNN Family
- **Process:**
  1. Generates region proposals in the image.
  2. Classifies each region using a convolutional neural network.
- **Variants:**
  - Fast R-CNN
  - Faster R-CNN
  - Mask R-CNN
- **Advantages:**
  - High accuracy due to region-based processing.
  - Effective for detecting small objects.
- **Disadvantages:**
  - Computationally expensive.
  - Slower processing speed.
- **Example:**
  - **BMW** uses Faster R-CNN for traffic sign recognition, achieving **98% accuracy** in detecting small signs.
    - [BMW AI Research, 2025](https://www.bmwgroup.com/en/innovation/ai-research.html)

#### 2. YOLO Family
- **Process:**
  - Single-shot detection: Predicts classes and bounding boxes in one pass through the network.
- **Variants:**
  - YOLOv3, YOLOv4, YOLOv5, YOLOv6, YOLOv7, YOLOv8
- **Advantages:**
  - Real-time speed.
  - Efficient and fast detection suitable for real-time applications.
- **Disadvantages:**
  - Slightly lower accuracy compared to R-CNN.
- **Example:**
  - **Tesla** uses YOLOv5 for real-time pedestrian detection, achieving **95% accuracy** with **30 ms latency**.
    - [Tesla AI Day, 2025](https://www.tesla.com/AIDay)

---

### Trade-off: Accuracy (R-CNN) vs Real-time Speed (YOLO)

| **Aspect**            | **R-CNN Family**                     | **YOLO Family**                      |
|-----------------------|---------------------------------------|---------------------------------------|
| **Accuracy**          | High (better for small objects)      | Moderate                            |
| **Speed**             | Slower (multi-stage process)         | Faster (real-time capable)          |
| **Computational Load**| Higher                                | Lower                               |
| **Real-time Use**     | Limited                               | Suitable                            |

---


### Image: Object Detection Approaches

![Object Detection Approaches](https://via.placeholder.com/800x400?text=Object+Detection+Approaches)
<small>Illustration of the R-CNN and YOLO approaches for object detection in images. R-CNN focuses on region-based detection with high accuracy, while YOLO emphasizes speed and real-time performance.</small>


### Key Points to Remember

- **Core Message:**
  - "Object detection using YOLO and R-CNN provides essential capabilities for autonomous driving and ADAS (Advanced Driver-Assistance Systems), balancing accuracy and speed."
  - Real-time processing capability of YOLO.
- **Call to Action:**
  - "Adopt a tailored approach to object detection, leveraging R-CNN for accuracy-critical tasks and YOLO for real-time applications to enhance safety and performance in automotive systems."

---
![Object Detection Illustration](https://via.placeholder.com/800x400?text=YOLO+vs+R-CNN+Object+Detection)
<small>This image illustrates the difference between YOLO and R-CNN approaches, highlighting their respective strengths in terms of speed and accuracy.</small>
