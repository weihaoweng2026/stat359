import argparse
import json
import random
import time
import os
from openai import OpenAI  

PERSONAS = {
    "Pirate": "a rough sea-captain, using 'Arrr', 'matey', and nautical terms.",
    "Shakespearean": "a 16th-century playwright, using 'thou', 'thee', and poetic metaphors.",
    "Technical Writer": "a precise manual author, using numbered steps, clear headings, and objective tone."
}

TOPICS = [
    "friendship", "adventure", "finding a lost treasure", "making a new invention", "bravery",
    "sharing a snack", "learning to fly", "fixing a broken toy", "a magical birthday", "solving a mystery",
    "helping a neighbor", "overcoming fear", "planting a seed", "visiting the moon", "a rainy afternoon"
]
THEMES = [
    "a hidden forest", "a futuristic city", "a stormy ocean", "a quiet library", 
    "a candy castle", "a dusty attic", "a busy marketplace", "a colorful garden",
    "a snowy mountain", "a tiny mouse hole", "a giant's kitchen", "a submarine"
]
INITIALS = [
    "It was a sunny day.", "Suddenly, a bell rang.", "Everyone was looking at the sky.",
    "The wind started to blow hard.", "I found a strange key.", "Once there was a small cat.",
    "The clock struck midnight.", "Nobody knew what happened next."
]

def generate_prompt(persona_name, topic, theme, initial):
    persona_desc = PERSONAS[persona_name]
    return (
        f"Write a very short story (max 150 words) for children using simple language.\n"
        f"Persona: Speak like {persona_desc}\n"
        f"Topic: {topic}\n"
        f"Setting: {theme}\n"
        f"Constraint: Start the story exactly with: '{initial}'\n"
        f"Format: Provide only the story text."
    )

def main():
    parser = argparse.ArgumentParser(description="Generate SimpleStory Dataset")
    parser.add_argument("--api_key", type=str, required=True, help="Your DeepSeek API Key")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to generate per persona")
    parser.add_argument("--output_file", type=str, default="simplestory_text/my_simplestories.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    client = OpenAI(
        api_key=args.api_key, 
        base_url="https://api.deepseek.com" 
    )

    with open(args.output_file, "a", encoding="utf-8") as f:
        for persona in PERSONAS.keys():
            print(f"\n--- Generating for Persona: {persona} ---")
            
            for i in range(args.num_samples):
                topic = random.choice(TOPICS)
                theme = random.choice(THEMES)
                initial = random.choice(INITIALS)
                prompt = generate_prompt(persona, topic, theme, initial)

                try:
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=1.3,
                        max_tokens=300
                    )
                    story_content = response.choices[0].message.content.strip()
                    entry = {
                        "text": f"<user> Write a story as a {persona} about {topic}. <assistant> {story_content}",
                        "metadata": {"persona": persona, "topic": topic, "theme": theme}
                    }
                    
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    f.flush()
                    
                    if (i + 1) % 5 == 0:
                        print(f"[{persona}] Progress: {i+1}/{args.num_samples} stories saved.")
                    
                except Exception as e:
                    print(f" [Error] Sample {i+1} failed: {e}")
                    time.sleep(5)
                    continue
                
                time.sleep(0.1)

    print(f"\n Done! Dataset saved to: {args.output_file}")

if __name__ == "__main__":
    main()