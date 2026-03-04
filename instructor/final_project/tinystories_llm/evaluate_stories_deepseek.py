import torch
import json
import argparse
import os
import pickle
import re
from openai import OpenAI
from tqdm import tqdm
from transformer_model import TinyStoriesConfig, TinyStoriesForCausalLM

class BPETokenizer:
    def __init__(self, vocab_size=10000, special_tokens=None):
        self.vocab_size = vocab_size
        self.token2id = {}
        self.id2token = {}
        self.bpe_codes = {}

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        tokenizer = cls(vocab_size=state['vocab_size'])
        tokenizer.token2id = state['token2id']
        tokenizer.id2token = state['id2token']
        tokenizer.bpe_codes = state['bpe_codes']
        return tokenizer

    def encode(self, text, add_special_tokens=False):
        tokens = text.strip().split()
        return [self.token2id.get(t, 0) for t in tokens]

    def decode(self, ids):
        return " ".join([self.id2token.get(i, '<unk>') for i in ids])

def load_model(model_path, tokenizer_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    
    config = TinyStoriesConfig(
        vocab_size=state_dict['transformer.embeddings.word_embeddings.weight'].shape[0],
        hidden_size=256, num_hidden_layers=4, num_attention_heads=8,
        max_position_embeddings=state_dict['transformer.embeddings.position_embeddings.weight'].shape[0]
    )
    model = TinyStoriesForCausalLM(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    tokenizer = BPETokenizer.load(tokenizer_path)
    return model, tokenizer

def deepseek_judge(api_key, story, persona, topic):
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    
    system_prompt = (
        "You are a strict literary judge. Evaluate the story based on 3 criteria: \n"
        "1. Persona Consistency (How well it mimics the requested style)\n"
        "2. Instruction Following (Topic relevance)\n"
        "3. Narrative Coherence (Logical flow)\n\n"
        "You MUST provide scores in the following format exactly:\n"
        "Persona Score: X/10\n"
        "Instruction Score: X/10\n"
        "Coherence Score: X/10\n"
        "Overall Score: X/10\n"
        "Reason: [Your brief reason]"
    )
    
    user_content = f"Target Persona: {persona}\nTarget Topic: {topic}\nStory Content: {story}"
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def parse_scores(raw_response):
    scores = {
        "persona": 0,
        "instruction": 0,
        "coherence": 0,
        "overall": 0
    }
    patterns = {
        "persona": r"Persona Score:\s*(\d+)",
        "instruction": r"Instruction Score:\s*(\d+)",
        "coherence": r"Coherence Score:\s*(\d+)",
        "overall": r"Overall Score:\s*(\d+)"
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, raw_response, re.IGNORECASE)
        if match:
            scores[key] = int(match.group(1))
            
    return scores

def generate_text(model, tokenizer, prompt, device, max_len=200):
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=max_len,
            temperature=0.7,
            top_p=0.9
        )
    return tokenizer.decode(output_ids[0].tolist())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--simple_ckpt", type=str, required=True)
    parser.add_argument("--tiny_ckpt", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simple_model, tokenizer = load_model(args.simple_ckpt, args.tokenizer_path, device)
    tiny_model, _ = load_model(args.tiny_ckpt, args.tokenizer_path, device)

    topics = ["a magic mirror", "a robot's first day", "a brave hamster"]
    personas = ["Pirate", "Shakespearean", "Technical Writer"]
    results = []

    for topic in topics:
        print(f"\nEvaluating Topic: {topic}")
        for p in personas:
            prompt = f"<user> Write a story as a {p} about {topic}. <assistant>"
            story = generate_text(simple_model, tokenizer, prompt, device)
            clean_story = story.split("<assistant>")[-1].strip() if "<assistant>" in story else story
            
            raw_judge_msg = deepseek_judge(args.api_key, clean_story, p, topic)
            extracted_scores = parse_scores(raw_judge_msg)
            
            results.append({
                "model": "SimpleStory",
                "persona": p,
                "topic": topic,
                "raw_response": raw_judge_msg,
                "scores": extracted_scores,
                "story": clean_story
            })

        tiny_prompt = f"Once upon a time, there was {topic}."
        story_tiny = generate_text(tiny_model, tokenizer, tiny_prompt, device)
        raw_judge_tiny = deepseek_judge(args.api_key, story_tiny, "Standard", topic)
        extracted_scores_tiny = parse_scores(raw_judge_tiny)
        
        results.append({
            "model": "TinyStory",
            "persona": "Standard",
            "topic": topic,
            "raw_response": raw_judge_tiny,
            "scores": extracted_scores_tiny,
            "story": story_tiny
        })

    with open("detailed_comparison_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print("\nDetailed Report Saved with sub-scores!")

if __name__ == "__main__":
    main()