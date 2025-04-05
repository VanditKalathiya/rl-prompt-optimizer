import os
os.environ["TORCH_DISABLE_GPU"] = "1"  # optional: disables GPU inference if it's buggy
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # fixes some CPU issues

import streamlit as st
import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

from stable_baselines3 import PPO, A2C
from rl_env.prompt_env import PromptOptimizationEnv
from gymnasium import spaces
from utils.reward_function import clarity_score, redundancy_penalty, simulated_llm_rating, get_sentiment_score, compute_reward
from utils.groq_mock import call_groq_llm

# Load models
MODEL_PATHS = {
    "PPO": "saved_models/ppo_prompt_optimizer",
    "A2C": "saved_models/a2c_prompt_optimizer"
}

@st.cache_resource
def load_model(agent_type):
    if agent_type == "PPO":
        return PPO.load(MODEL_PATHS["PPO"])
    else:
        return A2C.load(MODEL_PATHS["A2C"])

st.title("🧠 RL Prompt Optimizer")

# Sidebar
st.sidebar.header("Agent Settings")
agent_type = st.sidebar.selectbox("Choose RL Agent", ["PPO", "A2C"])
show_breakdown = st.sidebar.checkbox("Show Reward Breakdown", value=True)

# User input
user_prompt = st.text_input("✍️ Enter your prompt:", "")

if user_prompt:
    if st.button("🚀 Generate Response"):
        prompt_pool = [user_prompt]
        env = PromptOptimizationEnv(prompt_pool)
        env.prompt_pool = prompt_pool
        env.observation_space = spaces.Discrete(len(prompt_pool))

        model = load_model(agent_type)
        obs, _ = env.reset()
        action, _ = model.predict(obs)
        _, reward, done, truncated, _ = env.step(action)

        modified_prompt = env.modify_prompt(user_prompt, action)
        original_response = call_groq_llm(user_prompt)
        optimized_response = call_groq_llm(modified_prompt)
        simulated_response = optimized_response  # still same as optimized for now
        st.markdown("### 🔁 Modified Prompt")
        st.info(modified_prompt)

        st.markdown("### 🤖 LLM Responses")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("🧾 Original")
            st.success(original_response)
        with col2:
            st.subheader("🧠 Optimized")
            st.success(optimized_response)

        st.markdown(f"### 🏆 Reward: `{reward:.4f}`")

        if show_breakdown:
            from utils.reward_function import clarity_score, redundancy_penalty, simulated_llm_rating, get_sentiment_score
            clarity = clarity_score(user_prompt, simulated_response)
            redundancy = redundancy_penalty(simulated_response)
            groq = simulated_llm_rating(modified_prompt)
            sentiment = get_sentiment_score(simulated_response)

            st.markdown("### 📊 Reward Breakdown")
            st.write(f"• Clarity (cosine similarity): `{clarity:.3f}`")
            st.write(f"• Redundancy penalty: `{redundancy:.3f}`")
            st.write(f"• Groq-style rating: `{groq:.3f}`")
            st.write(f"• Sentiment score: `{sentiment:.3f}`")

