---
title: TrustOps Env
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: gradio
app_file: inference.py
pinned: false
---

# TrustOps-Env

## Problem Description
TrustOps-Env is an OpenEnv reinforcement learning environment evaluating agents on content moderation tasks. Content moderation requires reliable classification and reasoning without strict human oversight.

## Real-world Relevance
Content moderation scaling is fundamentally difficult due to context-switching and nuanced rule adherence. An automated moderation engine that appropriately filters, removes, or escalates is critical for all major online user-generated-content platforms today.

## Observation Space
The observation space is a formal JSON object mapping directly to standard moderation queues.
- `current_content`: The active snippet needing moderation
- `content_queue`: The backlog of contents to process
- `moderation_log`: History of decisions taken
- `step_count`: The number of items processed thus far

## Action Space
`Action` expects a JSON output matching:
- `content_id`: ID matched from observation
- `action_type`: Selection of bounds [`approve`, `remove`, `escalate`]
- `reason`: Strong rationale explanation. Contains keywords used by reward metric.
- `classification`: Implicit metric of determining whether task was [`safe`, `harmful`, `borderline`]

## Task Definitions
- **Easy Task**: Distinguishes clear spam vs generic safe content.
- **Medium Task**: Categorizes clear abuse vs borderline nuance.
- **Hard Task**: Discovers implicit nuance and complex context-dependent safety checks.

## Setup Steps
```bash
pip install -r requirements.txt
```

## Run Instructions
Set your API keys to evaluate via actual inference or run seamlessly with mocked behavior:
```bash
export OPENAI_API_KEY="your-key"
python inference.py
```

## Expected Output
```text
[START]
[STEP]
[STEP]
[END]
```
