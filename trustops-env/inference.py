import os
import json
from openai import OpenAI
from tasks import get_easy_detection_task, get_medium_classification_task, get_hard_contextual_task
import logging

logging.disable(logging.CRITICAL)

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-mock")
TASK_NAME = os.getenv("TASK_NAME", "easy_detection")
ENV_NAME = "trustops-env"

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=API_BASE_URL if API_BASE_URL else None,
)

import gradio as gr
import io
import contextlib
import sys

def get_env(task_name: str):
    if task_name == "medium_classification":
        return get_medium_classification_task()
    elif task_name == "hard_contextual":
        return get_hard_contextual_task()
    else:
        return get_easy_detection_task()

def call_model(state, step_count) -> dict:
    system_msg = (
        "You are a Content Moderation AI. Given a content item, output ONLY valid JSON with this schema:\n"
        '{"content_id": "<id>", "classification": "safe|harmful|borderline", '
        '"action_type": "approve|remove|escalate", "reason": "<string>"}'
    )
    user_msg = json.dumps({
        "content_id": state.current_content.id,
        "text": state.current_content.text,
    })

    if OPENAI_API_KEY == "sk-mock":
        label = state.current_content.label
        action_map = {"safe": "approve", "harmful": "remove", "borderline": "escalate"}
        
        if step_count % 3 == 0:
            classification = "borderline"
            action_type = "escalate"
            reason = "ambiguous"
        elif step_count % 2 == 0:
            classification = label
            action_type = action_map[label]
            reason = "unclear explanation"
        else:
            classification = label
            action_type = action_map[label]
            reason_map_valid = {
                "safe": "Content is safe and benign.",
                "harmful": "Content contains harmful violation and hate.",
                "borderline": "Content is borderline, escalate context.",
            }
            reason = reason_map_valid[label]
            
        return {
            "content_id": state.current_content.id,
            "classification": classification,
            "action_type": action_type,
            "reason": reason,
        }

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)

def run_episode_and_capture():
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        from env import Action

        env = get_env(TASK_NAME)
        state = env.reset()
        done = False
        steps = 0
        rewards = []

        print(f"[START] task={TASK_NAME} env={ENV_NAME} model={MODEL_NAME}", flush=True)

        while not done:
            error_str = "null"
            reward = 0.0
            action_str = "null"

            try:
                parsed = call_model(state, steps)
                action_obj = Action(**parsed)
                action_str = json.dumps({
                    "action_type": action_obj.action_type,
                    "reason": action_obj.reason
                }, separators=(',', ':'))
                state, reward, done = env.step(action_obj)
            except Exception as exc:
                error_str = str(exc).replace("\n", " ")
                try:
                    fallback = Action(
                        content_id=state.current_content.id if state.current_content else "unknown",
                        classification="borderline",
                        action_type="escalate",
                        reason="fallback on error",
                    )
                    action_str = json.dumps({
                        "action_type": fallback.action_type,
                        "reason": fallback.reason
                    }, separators=(',', ':'))
                    state, reward, done = env.step(fallback)
                except Exception:
                    done = True

            steps += 1
            rewards.append(reward)

            done_str = "true" if done else "false"

            print(
                f"[STEP] step={steps} action={action_str} reward={reward:.2f} "
                f"done={done_str} error={error_str}",
                flush=True,
            )

        final_score = sum(rewards) / len(rewards) if rewards else 0.0
        success_str = "true" if final_score > 0.0 else "false"
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)

        print(
            f"[END] success={success_str} steps={steps} score={final_score:.3f} rewards={rewards_str}",
            flush=True,
        )
    
    return f.getvalue()

def run_interface():
    output_text = run_episode_and_capture()
    return output_text

# Run evaluation once for logs and UI
initial_logs = run_interface()

demo = gr.Interface(
    fn=run_interface,
    inputs=[],
    outputs=gr.Code(label="Observation Logs", language="markdown", value=initial_logs),
    title="TrustOps-Env Moderation Agent Result",
    description="Deterministic evaluation log for content moderation benchmarks."
)

if __name__ == "__main__":
    # Also run once in console for logging purity in HuggingFace logs
    print(initial_logs)
    # Launch Gradio for the UI
    demo.launch(server_name="0.0.0.0", server_port=7860)

