import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from utils.reward_function import compute_reward
from utils.groq_mock import call_groq_llm


class PromptOptimizationEnv(gym.Env):
    def __init__(self, prompt_pool):
        super(PromptOptimizationEnv, self).__init__()

        self.prompt_pool = prompt_pool  # List of prompts
        self.action_space = spaces.Discrete(5)  # 5 prompt modification strategies
        self.observation_space = spaces.Discrete(len(prompt_pool))  # Index of current prompt

        self.current_prompt = None
        self.state_index = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state_index = random.randint(0, len(self.prompt_pool) - 1)
        self.current_prompt = self.prompt_pool[self.state_index]
        return self.state_index, {}  # return observation and info (Gymnasium format)

    def step(self, action):
        # Modify prompt based on the action
        modified_prompt = self.modify_prompt(self.current_prompt, action)

        # 🔁 Call real local LLM for response
        response = self.fake_llm(modified_prompt)

        # Compute reward from full evaluation pipeline
        reward = self.evaluate_response(response)

        done = True  # One-step episodes
        return self.state_index, reward, done, False, {}

    def modify_prompt(self, prompt, action):
        strategies = [
            lambda p: p,  # No change
            lambda p: p + " please",  # Add politeness
            lambda p: "In detail, " + p,  # Add specificity
            lambda p: p + " (keep it short)",  # Length constraint
            lambda p: "What do you mean by: " + p  # Clarification-style
        ]
        return strategies[action](prompt)

    def fake_llm(self, prompt):
        # 🔁 Now using flan-t5-small instead of fake string
        return call_groq_llm(prompt)

    def evaluate_response(self, response):
        return compute_reward(self.current_prompt, response)


