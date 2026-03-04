import torch
import argparse
import os
import json
import pickle
from transformer_model import TinyStoriesConfig, TinyStoriesForCausalLM

class BPETokenizerWrapper:
    """Wrapper to ensure compatibility with the trained BPE tokenizer."""
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

def main():
    parser = argparse.ArgumentParser(description="Generate stories using the trained SimpleStory Chat model.")
    parser.add_argument('--model_path', type=str, default='simplestory_model/final_chat_model.pth', help='Path to model checkpoint')
    parser.add_argument('--tokenizer_path', type=str, default='simplestory_model/bpe_tokenizer_simplestories.pkl', help='Path to BPE tokenizer')
    parser.add_argument('--persona', type=str, default='Pirate', choices=['Pirate', 'Shakespearean', 'Technical Writer'], help='Persona for the AI')
    parser.add_argument('--topic', type=str, default='a brave small dog', help='Topic of the story')
    parser.add_argument('--max_length', type=int, default=150, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature (0.1-1.5)')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p (nucleus) sampling')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Device to use')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'auto' else ('cuda' if args.device == 'cuda' else 'cpu'))

    tokenizer = BPETokenizerWrapper.load(args.tokenizer_path)
    vocab_size = len(tokenizer.token2id)

    config_dir = os.path.dirname(args.model_path)
    config_path = os.path.join(config_dir, 'args.json')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            t_args = json.load(f)
        config = TinyStoriesConfig(
            vocab_size=vocab_size,
            hidden_size=t_args.get('hidden_size', 256),
            num_hidden_layers=t_args.get('num_layers', 4),
            num_attention_heads=t_args.get('num_heads', 8),
            max_position_embeddings=t_args.get('max_seq_len', 256)
        )
    else:
        config = TinyStoriesConfig(
            vocab_size=vocab_size,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            max_position_embeddings=256
        )

    model = TinyStoriesForCausalLM(config)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    prompt = f"<user> Write a story as a {args.persona} about {args.topic}. <assistant>"
    
    input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=True)], dtype=torch.long).to(device)
    eos_token_id = tokenizer.token2id.get('<eos>', None)

    print(f"\n[System] Character: {args.persona} | Topic: {args.topic}")
    print("-" * 50)
   
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_token_id=eos_token_id,
        )

    full_output = tokenizer.decode(output_ids[0].tolist())
    if "<assistant>" in full_output:
        response = full_output.split("<assistant>")[-1].strip()
    else:
        response = full_output

    print(f"Story: {response}")
    print("-" * 50)

if __name__ == '__main__':
    main()