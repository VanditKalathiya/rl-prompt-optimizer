from utils.sentiment_analysis import get_sentiment_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

def clarity_score(original_prompt, response):
    """
    Measures how similar the response is to an ideal clarified version.
    For now, simulate ideal response similarity with TF-IDF.
    """
    vectorizer = TfidfVectorizer().fit([original_prompt, response])
    tfidf_matrix = vectorizer.transform([original_prompt, response])
    sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return sim  # [0, 1]

def redundancy_penalty(response):
    """
    Penalize if too many words are repeated.
    Simple approach: count unique vs total words.
    """
    words = response.split()
    if len(words) == 0:
        return 0
    unique = len(set(words))
    redundancy = 1 - (unique / len(words))  # closer to 1 = more redundant
    return redundancy

def simulated_llm_rating(prompt):
    """
    Simulates LLM clarity rating in response to:
    "Rate the clarity of this prompt on a scale of 1-10."
    """
    # Simple heuristic mock-up:
    if len(prompt) < 20:
        return random.uniform(4, 7)
    elif "please" in prompt or "in detail" in prompt or "keep it short" in prompt:
        return random.uniform(8, 10)
    else:
        return random.uniform(6, 8)


def compute_reward(original_prompt, response, λ1=1.0, λ2=0.7, λ3=0.5, α=0.6):
    sim_score = clarity_score(original_prompt, response)
    redundancy = redundancy_penalty(response)
    llm_rating = simulated_llm_rating(original_prompt)  # ← This is our Groq-style score
    sentiment = get_sentiment_score(response)

    reward = λ1 * sim_score - λ2 * redundancy + λ3 * llm_rating + α * sentiment
    return reward
