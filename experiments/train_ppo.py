import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl_env.prompt_env import PromptOptimizationEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from rl_env.prompt_env import PromptOptimizationEnv

# Sample prompts for training
prompt_pool = [
    "What is the capital of France?",
    "Explain quantum physics in simple terms.",
    "Tell me a joke about AI.",
    "How to cook vegetarian pasta?",
    "What's the weather like tomorrow in Toronto?"
]

# Create environment
env = PromptOptimizationEnv(prompt_pool)

# Optional: check environment sanity
check_env(env, warn=True)

# Initialize PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=5000)

# Save the trained model
model.save("D:/IOT/rl_prompt_optimizer/saved_models/ppo_prompt_optimizer")

print("✅ PPO training completed and model saved.")
