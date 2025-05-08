#!/usr/bin/env python3
import argparse
import torch
import os

from transformerModel2 import Transformer
from pcap_tokenization2 import TextTokenizer, PacketTokenizer, decode_tokens_to_packets

def load_tokenizers_from_checkpoint(checkpoint):
    text_tokenizer_dict = checkpoint.get('text_tokenizer')
    packet_tokenizer_dict = checkpoint.get('packet_tokenizer')
    
    text_tokenizer = TextTokenizer()
    
    text_tokenizer.word2id = text_tokenizer_dict
    text_tokenizer.token_to_id = text_tokenizer_dict
    text_tokenizer.vocab_size = len(text_tokenizer_dict)
    
    # Build packet tokenizer
    packet_tokenizer = PacketTokenizer()
    packet_tokenizer.token2id = packet_tokenizer_dict
    packet_tokenizer.id2token = {v: k for k, v in packet_tokenizer_dict.items()}
    packet_tokenizer.vocab_size = len(packet_tokenizer_dict)
    
    return text_tokenizer, packet_tokenizer

def generate_packets(model, input_text, text_tokenizer, packet_tokenizer, device, max_len=100, k=5):
    """
    Generate output tokens for a given prompt using the model.
    """
    model.eval()
    with torch.no_grad():
        # Encode input prompt using text tokenizer
        src = text_tokenizer.encode(input_text)
        src_tensor = torch.tensor([src], dtype=torch.long).to(device)
        
        if "<STREAM_START>" in packet_tokenizer.token2id:
            trg_indices = [packet_tokenizer.token2id["<STREAM_START>"]]
        else:
            trg_indices = []
        
        # Generate tokens one by one
        for _ in range(max_len):
            trg_tensor = torch.tensor([trg_indices], dtype=torch.long).to(device)
            output = model(src_tensor, trg_tensor)
            # Get logits from the last step
            logits = output[:, -1, :]
            
            k_val = k
            values, _ = torch.topk(logits, k_val)
            min_values = values[:, -1].unsqueeze(1)
            
            top_k_filtered_logits = torch.where(logits < min_values.to(logits.dtype),
                                                float('-inf'),
                                                logits)
            probabilities = torch.softmax(top_k_filtered_logits, dim=1)
            next_token = torch.multinomial(probabilities, 1).item()
            
            # If the EOS token is generated, stop.
            if "<EOS>" in packet_tokenizer.token2id and next_token == packet_tokenizer.token2id["<EOS>"]:
                break
            
            trg_indices.append(next_token)
        
        if "<STREAM_END>" in packet_tokenizer.token2id:
            trg_indices.append(packet_tokenizer.token2id["<STREAM_END>"])
        
        return packet_tokenizer.decode(trg_indices)

def main():
    parser = argparse.ArgumentParser(description="Load a saved Transformer model and test it with a prompt.")
    parser.add_argument("--checkpoint", type=str, default="best_model.pth", help="Path to the checkpoint file")
    parser.add_argument("--prompt", type=str, help="Text prompt to test the model")
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint file {args.checkpoint} not found!")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    
    # checkpoint = torch.load(args.checkpoint, map_location=device)
    # checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))

    text_tokenizer, packet_tokenizer = load_tokenizers_from_checkpoint(checkpoint)
    
    src_vocab_size = text_tokenizer.vocab_size
    trg_vocab_size = packet_tokenizer.vocab_size
    src_pad_idx = text_tokenizer.word2id.get("<PAD>")
    trg_pad_idx = packet_tokenizer.token2id.get("<PAD>")
    
    embed_size = 512
    num_layers = 6
    forward_expansion = 6
    heads = 8
    dropout = 0.1
    max_length = 1024
    
    model = Transformer(
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        embed_size=embed_size,
        num_layers=num_layers,
        forward_expansion=forward_expansion,
        heads=heads,
        dropout=dropout,
        device=device,
        max_length=max_length
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)  # Moves the model to GPU if available

    print("Model loaded successfully from checkpoint")
    
    if args.prompt:
        user_prompt = args.prompt
    else:
        user_prompt = input("Enter your prompt: ")
    
    generated_tokens = generate_packets(model, user_prompt, text_tokenizer, packet_tokenizer, device, max_len=100, k=5)
    print("\nGenerated tokens:")
    print(generated_tokens)
    
    decoded_packets = decode_tokens_to_packets(generated_tokens)
    print("\nDecoded packets:")
    for idx, pkt in enumerate(decoded_packets):
        print(f"Packet {idx + 1}: {pkt}")

if __name__ == "__main__":
    main()
# python3 test_model1.py --checkpoint model_checkpoint.pth
