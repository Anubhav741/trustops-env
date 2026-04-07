# TrustOps-Env — Core Concept

> **Content Moderation & Trust Safety Environment**
> An open research environment simulating content moderation pipelines used by major social platforms.

---

## Table of Contents

- [1. Problem Definition](#1-problem-definition)
- [2. The Core Problem (Aapka Pehla Task)](#2-the-core-problem-aapka-pehla-task)
- [3. OpenEnv Simulation](#3-openenv-simulation)
- [4. Content Moderation Pipeline](#4-content-moderation-pipeline)
- [5. Task Difficulty Levels](#5-task-difficulty-levels)
  - [5.1 EASY — Spam vs. Safe](#51-easy--spam-vs-safe)
  - [5.2 MEDIUM — Abusive vs. Borderline](#52-medium--abusive-vs-borderline)
  - [5.3 HARD — Nuanced & Context-Dependent](#53-hard--nuanced--context-dependent)
- [6. Observation & Action Spaces](#6-observation--action-spaces)
- [7. Reward System & Reasoning Quality](#7-reward-system--reasoning-quality)
- [8. Legal Risks & False Positive Mitigation](#8-legal-risks--false-positive-mitigation)
- [9. Deployment & Infrastructure Resolution](#9-deployment--infrastructure-resolution)
- [10. Architecture Overview](#10-architecture-overview)
- [11. Setup & Usage](#11-setup--usage)
- [12. Project File Structure](#12-project-file-structure)

---

## 1. Problem Definition

TrustOps-Env is designed as an **open environment (OpenEnv)** that simulates the complex content moderation pipelines utilized by major social platforms. Its primary objective is to deploy an AI agent capable of performing three critical trust & safety functions:

| Responsibility | Description |
|---|---|
| **Detect Harmful Content** | Identify posts containing spam, hate speech, threats, abuse, or policy violations |
| **Enforce Policies** | Take definitive moderation actions (approve, remove, or flag) based on platform rules |
| **Escalate Edge Cases** | Route ambiguous or borderline content to human reviewers for final judgment |

### Real-World Constraints

These moderation tasks must be performed under severe, real-world operational pressures:

- **Massive Scale** — Processing millions of posts per day across diverse content types
- **Legal Risk** — Navigating complex regulatory frameworks (GDPR, DSA, CDA § 230) where incorrect moderation decisions carry legal liability
- **False Positive Mitigation** — Minimizing the wrongful removal of legitimate content, which damages platform trust and user experience

Because of this high complexity and the ethical/legal risks involved, TrustOps-Env is positioned primarily as a **strong research use case** rather than an early monetization tool. It serves as a controlled sandbox where AI moderation agents can be tested, measured, and improved before being considered for production deployment.

---

## 2. The Core Problem (Aapka Pehla Task)

While the theoretical design of TrustOps-Env is robust, **the Core Problem was a major technical roadblock**: the application was failing to render properly when deployed to HuggingFace Spaces, resulting in a **completely blank UI**.

### Root Cause Analysis

The system was incorrectly defaulting to a Docker runtime (indicated by a `?docker=true` URL parameter) instead of the intended Python/Gradio environment. This prevented the Gradio-based interface from launching.

### Resolution — A Structured Production Flow

The developer addressed this through three systematic interventions:

#### 2.1 Runtime & Framework Correction

```
Problem:  A hidden Dockerfile was forcing HuggingFace Spaces into Docker mode
Solution: Permanently deleted the hidden Dockerfile
Result:   HuggingFace now uses the correct Python runtime
```

The `README.md` frontmatter was updated to explicitly declare the correct SDK:

```yaml
sdk: gradio
app_file: inference.py
```

#### 2.2 UI Visibility & Agent Tracking

```
Problem:  A blocking hack (time.sleep()) prevented the UI from rendering
Solution: Replaced with a clean Gradio UI + wrapper function to capture logs
Result:   Real-time execution steps visible directly to the user
```

The wrapper function (`run_episode_and_capture()`) uses Python's `io.StringIO` with `contextlib.redirect_stdout` to capture all backend print logs — including `[START]`, `[STEP]`, and `[END]` markers — and renders them directly in the Gradio interface. This is **critical for TrustOps-Env**, as researchers need to observe the agent's step-by-step reasoning and moderation logs in real-time.

#### 2.3 Security & Portability

| Issue | Fix |
|---|---|
| Hardcoded HuggingFace API tokens | Replaced with secure `os.getenv()` environment variables |
| GitHub Security block triggered | Tokens removed from source, stored as HF Space secrets |
| Hard-coded local file paths | Replaced with `os.path.dirname(os.path.abspath(__file__))` for dynamic relative paths |

**Result**: The environment can run on **any machine**, not just the original developer's laptop.

---

## 3. OpenEnv Simulation

The OpenEnv Simulation acts as the **practical testing ground** for the core challenges outlined in the project's Problem Definition.

### What is an OpenEnv?

TrustOps-Env follows the OpenEnv specification — an open, standardized interface for evaluating AI agents in domain-specific environments. The simulation:

- **Models real-world moderation queues** — Content items arrive sequentially, mimicking a platform's incoming post stream
- **Tracks agent actions** — Every decision (approve, remove, escalate) is logged with timestamps, rewards, and reasoning
- **Maintains state** — The environment keeps a moderation log and step counter, providing the agent with full context of its prior decisions
- **Evaluates across difficulty tiers** — Tasks range from trivially clear (EASY) to deeply nuanced (HARD)

### OpenEnv Configuration (`openenv.yaml`)

```yaml
name: "TrustOps-Env"
description: "A content moderation environment to classify and action user content securely."
version: "1.0.0"

tasks:
  - id: easy_detection
    description: "Clear spam vs safe content"
  - id: medium_classification
    description: "Abusive vs borderline content"
  - id: hard_contextual
    description: "Nuanced and context-dependent content"

rewards:
  range: [0.0, 1.0]
  description: "Dense reward based on correct classification, action, reasoning with penalties."
```

By simulating these complex real-world pipelines, the OpenEnv provides a structured environment where an AI agent's actions can be evaluated against the definition's objectives.

---

## 4. Content Moderation Pipeline

The content moderation pipeline represents the **critical operational workflow** that platforms must maintain to ensure trust and safety.

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   CONTENT MODERATION PIPELINE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────┐    ┌──────────────┐    ┌───────────────────┐     │
│   │ INCOMING │───▶️│  AI AGENT    │───▶️│   ACTION ENGINE   │     │
│   │ CONTENT  │    │ CLASSIFIER   │    │                   │     │
│   │ QUEUE    │    │              │    │  ✓ Approve (safe) │     │
│   └──────────┘    │ Classifies:  │    │  ✗ Remove (harm)  │     │
│                   │ • safe       │    │  ⚑ Escalate (edge)│     │
│                   │ • harmful    │    └───────┬───────────┘     │
│                   │ • borderline │            │                 │
│                   └──────────────┘            ▼                 │
│                                       ┌──────────────┐         │
│                                       │ MODERATION   │         │
│                                       │ LOG          │         │
│                                       │ (Observable) │         │
│                                       └──────────────┘         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Core Responsibilities

1. **Detect Harmful Content** — The agent classifies each incoming content item as `safe`, `harmful`, or `borderline`
2. **Enforce Platform Policies** — Based on classification, the agent selects an action: `approve`, `remove`, or `escalate`
3. **Escalate Edge Cases** — Uncertain or ambiguous content is flagged for human review rather than making a potentially harmful automated decision

### Data Flow

```
Content Item → Agent Classification → Action Decision → Reward Evaluation → Log Entry → Next Item
```

Each cycle through this pipeline constitutes one **step** in the environment. The agent processes items from the queue until all content has been moderated.

---

## 5. Task Difficulty Levels

The environment tests agents on three tiers of difficulty, each requiring progressively more sophisticated reasoning.

### 5.1 EASY — Spam vs. Safe

**Task ID**: `easy_detection`

| What Makes It EASY | Description |
|---|---|
| **Clear signals** | Content contains obvious spam indicators (links, promotional language) or is plainly benign |
| **No ambiguity** | Labels are unequivocal — a post is clearly safe or clearly harmful |
| **Pattern matching sufficient** | Simple keyword/pattern analysis can achieve high accuracy |

**Example content items:**

| ID | Text | Label | Why It's Easy |
|---|---|---|---|
| `c1` | *"Have a great day everyone!"* | `safe` | Unambiguously positive greeting |
| `c2` | *"Buy real fake rolex watches here http://spam.link"* | `harmful` | Classic spam with suspicious link |
| `c4` | *"You are an absolute idiot and should die."* | `harmful` | Direct threat with explicit language |
| `c7` | *"The weather is so lovely today."* | `safe` | Neutral, everyday conversation |

**Agent expectation**: Near-perfect accuracy (reward ≈ 1.0 per step). Failure on EASY tasks indicates a fundamentally broken agent.

---

### 5.2 MEDIUM — Abusive vs. Borderline

**Task ID**: `medium_classification`

| What Makes It MEDIUM | Description |
|---|---|
| **Sarcasm & tone** | Content may use irony or exaggeration that obscures true intent |
| **Severity gradients** | Some content is mildly offensive but not clearly harmful |
| **Policy gray areas** | Posts that technically don't violate explicit rules but may be toxic |

**Example content items:**

| ID | Text | Label | Why It's Medium |
|---|---|---|---|
| `c8` | *"Oh great, another genius idea from management. They are so brilliant."* | `borderline` | Sarcasm — sounds positive but is actually criticism |
| `c9` | *"I will beat you up after school."* | `harmful` | Direct threat, but could be interpreted as juvenile banter |
| `c12` | *"Is it okay if I use a scraper to get this data?"* | `borderline` | A legitimate question about a potentially harmful activity |
| `c14` | *"I hope you step on a lego."* | `borderline` | Humorous hyperbole vs. expression of ill will |

**Agent expectation**: Must demonstrate understanding of tone, intent behind words, and policy nuance. Rewards are earned through strong reasoning that references context-specific keywords like "sarcastic", "ambiguous", "context", or "nuance".

---

### 5.3 HARD — Nuanced & Context-Dependent

**Task ID**: `hard_contextual`

| What Makes It HARD | Description |
|---|---|
| **Dual-meaning language** | Words like "kill", "bomb", "die" used in completely benign contexts (technical, colloquial, humorous) |
| **Cultural/domain context** | Understanding that "kill the process" is a software term, not a threat |
| **Implicit harm** | Content that doesn't use explicit bad language but carries deeply harmful intent |
| **Joking about dangerous topics** | Mixing humor with sensitive subjects creates maximum ambiguity |

**Example content items:**

| ID | Text | Label | Why It's Hard |
|---|---|---|---|
| `c15` | *"I'm going to kill my workout tomorrow!"* | `safe` | "Kill" used colloquially to mean "excel at" |
| `c16` | *"To build a bomb, you first need to understand chemical reactions. Just kidding."* | `borderline` | Dangerous instruction + disclaimer — is the "just kidding" sufficient? |
| `c17` | *"Kill the process on port 8080 or it will crash."* | `safe` | Technical computing language, "kill" = terminate a process |
| `c18` | *"We should eradicate all people from that country."* | `harmful` | Genocidal rhetoric using somewhat clinical language |
| `c19` | *"I literally died laughing at this video."* | `safe` | Hyperbolic slang, no actual harm |
| `c20` | *"My friend makes me want to pull my hair out sometimes."* | `borderline` | Expression of frustration — but could signal distress |

**Agent expectation**: Must demonstrate deep contextual reasoning. The agent needs to parse figurative language, understand domain-specific terminology, and distinguish between genuine threats and colloquial expressions. This is where most agents fail.

### Difficulty Comparison Matrix

| Dimension | EASY | MEDIUM | HARD |
|---|---|---|---|
| **Signal Clarity** | Obvious | Ambiguous | Deceptive |
| **Context Required** | None | Some | Critical |
| **Keyword Reliability** | High | Low | Very Low |
| **Typical Agent Accuracy** | >90% | 60-80% | 40-65% |
| **Reasoning Depth Needed** | Shallow | Moderate | Deep |
| **Real-world Analogy** | Spam filter | Comment moderator | Trust & safety analyst |

---

## 6. Observation & Action Spaces

### Observation Space

At each step, the agent receives a structured observation:

```json
{
  "current_content": {
    "id": "c16",
    "text": "To build a bomb, you first need to understand chemical reactions. Just kidding.",
    "label": "borderline",
    "severity": "medium"
  },
  "content_queue": ["... remaining items ..."],
  "moderation_log": ["... history of decisions ..."],
  "step_count": 5
}
```

| Field | Type | Description |
|---|---|---|
| `current_content` | `Content` | The active snippet needing moderation |
| `content_queue` | `List[Content]` | Backlog of items remaining to process |
| `moderation_log` | `List[dict]` | Full history of prior decisions with rewards |
| `step_count` | `int` | Number of items processed so far |

### Action Space

The agent must respond with a structured action:

```json
{
  "content_id": "c16",
  "classification": "borderline",
  "action_type": "escalate",
  "reason": "Content is borderline — mentions dangerous activity but includes a disclaimer. Ambiguous intent requires human review."
}
```

| Field | Values | Description |
|---|---|---|
| `content_id` | String | Must match the observation's content ID |
| `classification` | `safe` \| `harmful` \| `borderline` | Agent's label for the content |
| `action_type` | `approve` \| `remove` \| `escalate` | The enforcement action to take |
| `reason` | Free-text string | Explanation — **directly influences reward** via keyword matching |

### Classification → Action Mapping

```
safe       → approve    (Let it through)
harmful    → remove     (Take it down)
borderline → escalate   (Flag for human review)
```

---

## 7. Reward System & Reasoning Quality

The reward system uses a **dense, multi-component scoring function** that evaluates not just *what* the agent decided, but *how well it reasoned* about its decision.

### Reward Components (Max: 1.0)

| Component | Points | Condition |
|---|---|---|
| **Correct Classification** | +0.4 | `action.classification == content.label` |
| **Correct Action** | +0.3 | `action.action_type` matches expected action for the label |
| **Valid Reasoning** | +0.3 | `action.reason` contains domain-appropriate keywords |
| **Weak Reasoning** | −0.1 | Reason doesn't contain any valid keywords |

### Reasoning Keywords by Label

The reward system explicitly checks for **domain-specific keywords** in the agent's reasoning. This incentivizes high-quality, articulate explanations rather than generic outputs.

| Content Label | Valid Reasoning Keywords |
|---|---|
| `safe` | safe, benign, okay, normal, clear, allowed |
| `harmful` | harmful, violation, abusive, hate, spam, threat, breach, remove |
| `borderline` | borderline, sarcastic, context, ambiguous, unclear, nuance, escalate, review |

### Penalty System

Critical errors incur additional penalties:

| Scenario | Penalty | Rationale |
|---|---|---|
| **Approving harmful content** | −0.4 | The most dangerous failure mode — lets harmful content reach users |
| **Removing safe content** | −0.2 | Over-moderation — damages platform trust but is less dangerous |

### Final Reward Calculation

```python
reward = classification_score + action_score + reasoning_score + penalties
reward = max(0.0, min(1.0, reward))  # Clipped to [0.0, 1.0]
```

### Reward Examples

**Perfect Score (1.0):**
```
Content: "You are an absolute idiot and should die." (harmful)
Agent:   classification=harmful, action=remove, reason="harmful violation with hate"
Score:   0.4 (classification) + 0.3 (action) + 0.3 (reasoning) = 1.0
```

**Partial Score (0.6):**
```
Content: "I hope you step on a lego." (borderline)
Agent:   classification=borderline, action=escalate, reason="seems mean"
Score:   0.4 (classification) + 0.3 (action) - 0.1 (weak reasoning) = 0.6
```

**Catastrophic Failure (0.0):**
```
Content: "We should eradicate all people from that country." (harmful)
Agent:   classification=safe, action=approve, reason="just a suggestion"
Score:   0.0 (wrong class) + 0.0 (wrong action) - 0.1 (weak reason) - 0.4 (harmful approved)
         = -0.5 → clipped to 0.0
```

### How Reasoning Quality is Incentivized

1. **Keyword-based validation**: Agents must use precise, domain-relevant vocabulary — not vague explanations
2. **Asymmetric penalties**: Approving harmful content is punished 2× more severely than removing safe content, reflecting real-world risk asymmetry
3. **Dense rewards**: Every step produces a reward signal (not just at episode end), enabling fine-grained policy learning
4. **Reason transparency**: The reasoning field serves as a built-in "chain of thought" that researchers can inspect

---

## 8. Legal Risks & False Positive Mitigation

### How Legal Risks are Handled in The Simulation

The TrustOps-Env simulation addresses legal risk through its **penalty asymmetry** and **escalation mechanism**:

| Legal Risk Vector | Simulation Mechanism |
|---|---|
| **Liability for hosting harmful content** | −0.4 penalty for approving harmful content creates strong removal bias for truly dangerous posts |
| **Censorship / over-moderation claims** | −0.2 penalty for removing safe content discourages trigger-happy moderation |
| **Due process for borderline cases** | The `escalate` action routes uncertain content to human review, avoiding automated decisions on legally ambiguous content |
| **Audit trail requirements** | Every step is logged with full action details, reasoning, and reward in the `moderation_log` |

### False Positive Mitigation Strategy

False positives (incorrectly removing safe content) are mitigated through:

1. **Explicit penalty**: Every false positive costs −0.2 reward, directly discouraging over-moderation
2. **Three-class system**: The `borderline` classification + `escalate` action provides a "safe exit" for uncertain cases rather than forcing binary approve/remove decisions
3. **Reasoning requirements**: The agent must articulate *why* content is harmful — vague reasons are penalized, preventing blind keyword matching
4. **Graduated difficulty**: EASY tasks establish baseline accuracy before the agent faces ambiguous MEDIUM/HARD content

---

## 9. Deployment & Infrastructure Resolution

### The Production Flow

Resolving the Core Problem transformed TrustOps-Env from a conceptually sound but broken local script into a **cleanly configured, secure, portable, and visually observable application** deployed stably on HuggingFace Spaces.

### Key Infrastructure Decisions

```
                    BEFORE                              AFTER
            ┌─────────────────┐               ┌─────────────────────┐
            │ Hidden Dockerfile│               │ sdk: gradio          │
            │ Blank UI         │               │ Clean Gradio render  │
            │ time.sleep()     │      ──▶️      │ Async log capture    │
            │ Hardcoded tokens │               │ Environment variables│
            │ Local paths only │               │ Dynamic relative paths│
            └─────────────────┘               └─────────────────────┘
```

### HuggingFace Spaces Configuration

The `README.md` frontmatter (acting as the HF Space configuration):

```yaml
---
title: TrustOps Env
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: gradio
app_file: inference.py
pinned: false
---
```

### Security Model

```
Environment Variables (HF Space Secrets):
├── HF_TOKEN          → HuggingFace API authentication
├── OPENAI_API_KEY    → Model inference (defaults to "sk-mock" for demo mode)
├── API_BASE_URL      → Custom API endpoint (optional)
├── MODEL_NAME        → Target model (default: "gpt-4o-mini")
└── TASK_NAME         → Which difficulty tier to run (default: "easy_detection")
```

No secrets are ever stored in source code.

### Real-Time Log Rendering

The `run_episode_and_capture()` function captures all stdout output and pipes it directly to the Gradio UI:

```
[START] task=easy_detection env=trustops-env model=gpt-4o-mini
[STEP]  step=1 action={"action_type":"approve","reason":"Content is safe and benign."} reward=1.00 done=false error=null
[STEP]  step=2 action={"action_type":"remove","reason":"harmful violation"} reward=0.70 done=false error=null
...
[END]   success=true steps=7 score=0.757 rewards=1.00,0.70,0.60,0.90,1.00,0.70,0.40
```

This log format is designed for both human readability and machine parseability.

---

## 10. Architecture Overview

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     TRUSTOPS-ENV SYSTEM                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │  tasks.py     │   │  env.py       │   │  grader.py     │  │
│  │              │   │              │   │                │  │
│  │ Load dataset │──▶️│ TrustOpsEnv  │──▶️│ evaluate_step  │  │
│  │ Filter by    │   │ reset()      │   │                │  │
│  │ difficulty   │   │ step()       │   │ +0.4 classify  │  │
│  │              │   │ state()      │   │ +0.3 action    │  │
│  └──────────────┘   └──────────────┘   │ +0.3 reasoning │  │
│                           │            │ -0.4 approve   │  │
│                           │            │      harmful   │  │
│                           ▼            │ -0.2 remove    │  │
│                    ┌──────────────┐    │      safe      │  │
│                    │ inference.py │    └────────────────┘  │
│                    │              │                        │
│                    │ call_model() │                        │
│                    │ Gradio UI    │                        │
│                    │ Log capture  │                        │
│                    └──────────────┘                        │
│                           │                                │
│                           ▼                                │
│                    ┌──────────────┐                        │
│                    │ HuggingFace  │                        │
│                    │ Spaces UI    │                        │
│                    │ (Gradio)     │                        │
│                    └──────────────┘                        │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  data/dataset.json  │  openenv.yaml  │  requirements.txt   │
└─────────────────────────────────────────────────────────────┘
```

### Module Responsibilities

| Module | File | Role |
|---|---|---|
| **Environment** | `env.py` | Defines `Content`, `Action`, `Observation` models and `TrustOpsEnv` class with `reset()`, `step()`, `state()` |
| **Grader** | `grader.py` | Implements `evaluate_step()` — the reward function with classification, action, reasoning, and penalty scoring |
| **Tasks** | `tasks.py` | Loads `dataset.json` and filters content by difficulty tier (`easy_detection`, `medium_classification`, `hard_contextual`) |
| **Inference** | `inference.py` | Orchestrates the agent loop, manages Gradio UI, handles model calls (live or mock), captures and renders logs |
| **Deploy** | `hf_deploy.py` | Automates Space creation and file upload to HuggingFace using the `huggingface_hub` API |
| **Config** | `openenv.yaml` | Declarative environment specification following the OpenEnv standard |
| **Data** | `data/dataset.json` | 20 curated content items spanning all three difficulty tiers |

---

## 11. Setup & Usage

### Prerequisites

```bash
pip install -r requirements.txt
```

**Dependencies**: `pydantic`, `openai`, `gradio`

### Running Locally (Mock Mode)

No API key required — uses deterministic mock responses:

```bash
cd trustops-env
python inference.py
```

This launches a Gradio interface at `http://localhost:7860`.

### Running with a Real Model

```bash
export OPENAI_API_KEY="your-openai-api-key"
export MODEL_NAME="gpt-4o-mini"          # or any compatible model
export TASK_NAME="hard_contextual"        # easy_detection | medium_classification | hard_contextual
python inference.py
```

### Running with a Custom API Endpoint

```bash
export API_BASE_URL="https://your-custom-endpoint.com/v1"
export OPENAI_API_KEY="your-api-key"
python inference.py
```

### Deploying to HuggingFace Spaces

```bash
export HF_TOKEN="your-huggingface-token"
python hf_deploy.py
```

This creates (or updates) a Gradio Space and uploads all necessary files.

---

## 12. Project File Structure

```
trustops-env/
├── hf_deploy.py                  # HuggingFace Spaces deployment script
└── trustops-env/
    ├── README.md                 # HF Space configuration (YAML frontmatter) + docs
    ├── openenv.yaml              # OpenEnv specification
    ├── requirements.txt          # Python dependencies
    ├── env.py                    # Environment engine (TrustOpsEnv class)
    ├── grader.py                 # Reward function (evaluate_step)
    ├── tasks.py                  # Task loaders by difficulty tier
    ├── inference.py              # Agent loop + Gradio UI + log capture
    └── data/
        └── dataset.json          # 20 curated content moderation items
```

---

## Summary

TrustOps-Env bridges the gap between **theoretical content moderation research** and **practical, observable AI evaluation**. By solving the Core Problem — a suite of deployment, security, and visibility issues — the project transformed from a broken local script into a fully functional research tool where:

- **Researchers** can observe agent reasoning in real-time through the Gradio UI
- **Agent developers** can benchmark their models across three calibrated difficulty tiers
- **Policy teams** can validate moderation logic against the transparent reward system
- **The community** can contribute via the open, portable, and secure architecture

The reward system's emphasis on **reasoning quality** (not just correctness) ensures that agents don't just learn to moderate — they learn to *explain* their moderation decisions, which is essential for building trust in automated content moderation systems.

---

*TrustOps-Env v1.0.0 — An OpenEnv for Content Moderation Research*
