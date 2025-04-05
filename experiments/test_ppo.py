import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from rl_env.prompt_env import PromptOptimizationEnv

# Sample prompt pool
prompt_pool = [
    "What is the capital of France?",
    "Explain quantum physics in simple terms.",
    "Tell me a joke about AI.",
    "How to cook vegetarian pasta?",
    "What's the weather like tomorrow in Toronto?"
]

# Initialize environment
env = PromptOptimizationEnv(prompt_pool)

# ✅ Load the trained model from the correct path
model = PPO.load("D:/IOT/rl_prompt_optimizer/saved_models/ppo_prompt_optimizer")

# Run a few test episodes
for i in range(5):
    obs, _ = env.reset()  # updated for gymnasium format
    action, _ = model.predict(obs)
    _, reward, done, truncated, _ = env.step(action)

    original_prompt = prompt_pool[obs]
    modified_prompt = env.modify_prompt(original_prompt, action)
    response = env.fake_llm(modified_prompt)

    print(f"\n🔹 Test {i+1}")
    print(f" Original Prompt: {original_prompt}")
    print(f" Modified Prompt: {modified_prompt}")
    print(f" Simulated Response: {response}")
    print(f" Reward: {reward:.4f}")
