import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

from gymnasium import spaces
from stable_baselines3 import PPO, A2C
from rl_env.prompt_env import PromptOptimizationEnv


# Load your trained model here: choose either PPO or A2C
MODEL_TYPE = "ppo"  # or "a2c"

MODEL_PATHS = {
    "ppo": "saved_models/ppo_prompt_optimizer",
    "a2c": "saved_models/a2c_prompt_optimizer"
}

prompt_pool = ["dummy"]  # We'll dynamically add user prompt

# Create environment
env = PromptOptimizationEnv(prompt_pool)

# Load model
if MODEL_TYPE == "ppo":
    model = PPO.load(MODEL_PATHS["ppo"])
else:
    model = A2C.load(MODEL_PATHS["a2c"])

print("\n🚀 RL Prompt Optimizer CLI")
print("Type your prompt below. Type 'exit' to quit.\n")

while True:
    user_input = input("📝 Your prompt: ").strip()
    if user_input.lower() == 'exit':
        break

    prompt_pool = [user_input]
    env.prompt_pool = prompt_pool  # Update environment with new prompt
    env.observation_space = spaces.Discrete(len(prompt_pool))

    obs, _ = env.reset()
    action, _ = model.predict(obs)
    _, reward, done, truncated, _ = env.step(action)

    original = prompt_pool[obs]
    modified = env.modify_prompt(original, action)
    response = env.fake_llm(modified)

    print("\n--- Response ---")
    print(f"📥 Original Prompt: {original}")
    print(f"🛠️ Modified Prompt: {modified}")
    print(f"💬 Response: {response}")
    print(f"🏆 Reward: {reward:.4f}")
    print("----------------\n")
