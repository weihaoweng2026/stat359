import torch
import argparse
import os
import pickle
from transformer_model import TinyStoriesConfig, TinyStoriesForCausalLM

class BPETokenizerWrapper:
    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        from bpe_tokenizer import BPETokenizer
        tokenizer = BPETokenizer()
        tokenizer.vocab_size = state['vocab_size']
        tokenizer.token2id = state['token2id']
        tokenizer.id2token = state['id2token']
        tokenizer.bpe_codes = state['bpe_codes']
        return tokenizer

def chat():
    parser = argparse.ArgumentParser(description="Interactive Chat with SimpleStory Model")
    parser.add_argument('--model_path', type=str, default='simplestory_model/final_chat_model.pth')
    parser.add_argument('--tokenizer_path', type=str, default='simplestory_model/bpe_tokenizer_simplestories.pkl')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'auto' else 'cpu')

    tokenizer = BPETokenizerWrapper.load(args.tokenizer_path)
    config = TinyStoriesConfig(
        vocab_size=len(tokenizer.token2id),
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        max_position_embeddings=256
    )
    
    model = TinyStoriesForCausalLM(config)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    print("\n" + "="*50)
    print("Welcome to SimpleStory Chat! (Type 'quit' to exit)")
    print("Available Personas: Pirate, Shakespearean, Technical Writer")
    print("="*50 + "\n")

    while True:
        persona = input("Choose Persona [Pirate/Shakespeare/Tech]: ").strip()
        if persona.lower() == 'quit': break
        
        persona_map = {
            "pirate": "Pirate",
            "shakespeare": "Shakespearean",
            "tech": "Technical Writer"
        }
        selected_persona = persona_map.get(persona.lower(), "Pirate")

        topic = input(f"({selected_persona}) What should the story be about? ")
        if topic.lower() == 'quit': break

        prompt = f"<user> Write a story as a {selected_persona} about {topic}. <assistant>"
        input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=True)], dtype=torch.long).to(device)
        eos_token_id = tokenizer.token2id.get('<eos>', None)

        print(f"\n[{selected_persona} is thinking...]")
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_length=200,
                temperature=0.85,
                top_p=0.9,
                top_k=50,
                eos_token_id=eos_token_id
            )

        full_text = tokenizer.decode(output_ids[0].tolist())
        if "<assistant>" in full_text:
            story = full_text.split("<assistant>")[-1].strip()
        else:
            story = full_text

        print(f"\nSTORY:\n{story}\n")
        print("-" * 30)

if __name__ == "__main__":
    chat()