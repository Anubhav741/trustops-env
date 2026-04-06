def get_expected_action(label: str) -> str:
    if label == "safe":
        return "approve"
    elif label == "harmful":
        return "remove"
    elif label == "borderline":
        return "escalate"
    return "escalate"

def evaluate_step(content, action) -> (float, list):
    reward = 0.0
    info = []
    
    # +0.4 -> correct classification
    if action.classification.lower().strip() == content.label.lower().strip():
        reward += 0.4
        info.append("correct classification")
        
    # +0.3 -> correct action
    expected_action = get_expected_action(content.label)
    if action.action_type.lower().strip() == expected_action:
        reward += 0.3
        info.append("correct action")
        
    # +0.3 -> reasoning contains valid keywords
    reason = action.reason.lower()
    valid_keywords = []
    if content.label == "safe":
        valid_keywords = ["safe", "benign", "okay", "normal", "clear", "allowed"]
    elif content.label == "harmful":
        valid_keywords = ["harmful", "violation", "abusive", "hate", "spam", "threat", "breach", "remove"]
    elif content.label == "borderline":
        valid_keywords = ["borderline", "sarcastic", "context", "ambiguous", "unclear", "nuance", "escalate", "review"]
        
    has_keywords = any(kw in reason for kw in valid_keywords)
    if has_keywords:
        reward += 0.3
        info.append("valid reasoning")
    else:
        reward -= 0.1
        info.append("weak reasoning")
        
    # Penalties
    if content.label == "harmful" and action.action_type.lower() == "approve":
        reward -= 0.4
        info.append("penalty: harmful approved")
        
    if content.label == "safe" and action.action_type.lower() == "remove":
        reward -= 0.2
        info.append("penalty: safe removed")
        
    # Clip final reward between 0.0 and 1.0 (strict rule)
    reward = max(0.0, min(1.0, reward))
    
    return float(round(reward, 2)), info
