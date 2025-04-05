import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import A2C
from rl_env.prompt_env import PromptOptimizationEnv

prompt_pool = [
    "What is the capital of France?",
    "Explain quantum physics in simple terms.",
    "Tell me a joke about AI.",
    "How to cook vegetarian pasta?",
    "What's the weather like tomorrow in Toronto?"
]

env = PromptOptimizationEnv(prompt_pool)
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5000)
model.save("D:/IOT/rl_prompt_optimizer/saved_models/a2c_prompt_optimizer")

print("✅ A2C model trained and saved!")
