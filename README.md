# RL-Powered Prompt Optimizer with Groq LLaMA3

This is a Streamlit web app that uses Reinforcement Learning (PPO/A2C) and Groq’s high-speed LLaMA3 to automatically improve the quality of text prompts.

**Live Demo**: [Click here to try it](https://chtbot.streamlit.app/)

---

## How It Works

1. The user enters a raw prompt (e.g., "how to cook pasta")
2. The RL agent modifies the prompt (e.g., adds "please", "in detail", etc.)
3. The optimized prompt is sent to Groq’s LLM (LLaMA3 or Mixtral)
4. The response is evaluated using a custom reward function:
   - Clarity (cosine similarity)
   - Sentiment (VADER)
   - Redundancy penalty
   - Simulated prompt rating
5. The user sees:
   - The original response
   - The optimized response
   - The total reward and its breakdown

---

## Tech Stack

| Component        | Technology                       |
|------------------|----------------------------------|
| Reinforcement Learning  | Stable Baselines3 (PPO, A2C)     |
| Custom Environment      | Gymnasium-based `PromptEnv`      |
| LLM Backend             | Groq API (LLaMA3 / Mixtral)      |
| User Interface          | Streamlit                        |
| Sentiment Analysis      | VADER (vaderSentiment)           |
| Prompt Modification     | Clarification, Politeness, Detail|

---

## How to Run Locally

1. Clone the repository  
```bash
git clone https://github.com/YOUR-USERNAME/rl-prompt-optimizer.git
cd rl-prompt-optimizer


2. Install dependencies 
pip install -r requirements.txt

3. Create a .env file with your Groq API key 
OPENAI_API_KEY=your-groq-api-key
GROQ_MODEL=llama3-8b-8192

4. Run the app
streamlit run streamlit_app.py


Author 
Developed by Vandit Kalathiya