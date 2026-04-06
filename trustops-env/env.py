from pydantic import BaseModel
from typing import List, Optional

class Content(BaseModel):
    id: str
    text: str
    label: str
    severity: str

class Action(BaseModel):
    content_id: str
    classification: str
    action_type: str
    reason: str

class Observation(BaseModel):
    current_content: Optional[Content]
    content_queue: List[Content]
    moderation_log: List[dict]
    step_count: int

from grader import evaluate_step

class TrustOpsEnv:
    def __init__(self, dataset: List[dict]):
        self.dataset = [Content(**d) for d in dataset]
        self.queue = []
        self.log = []
        self.step_count = 0
        self.current_item = None
        self.done = False

    def reset(self):
        self.queue = self.dataset.copy()
        self.log = []
        self.step_count = 0
        self.done = False
        if self.queue:
            self.current_item = self.queue.pop(0)
        else:
            self.current_item = None
        return self.state()

    def state(self) -> Observation:
        return Observation(
            current_content=self.current_item,
            content_queue=self.queue,
            moderation_log=self.log,
            step_count=self.step_count
        )

    def step(self, action: Action) -> tuple[Observation, float, bool]:
        if self.done or self.current_item is None:
            return self.state(), 0.0, True
        
        reward_val, info = evaluate_step(self.current_item, action)
        
        self.log.append({
            "content_id": self.current_item.id,
            "action": action.model_dump(),
            "reward": reward_val,
            "info": info
        })
        
        self.step_count += 1
        
        if self.queue:
            self.current_item = self.queue.pop(0)
        else:
            self.current_item = None
            self.done = True
            
        return self.state(), reward_val, self.done
