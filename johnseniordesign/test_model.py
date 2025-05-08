import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformerModel2 import Transformer
from train_model2 import generate_packets

from collections import Counter
import matplotlib.pyplot as plt

from pcap_tokenization2 import (
    MultiPacketStreamDataset, TextTokenizer, PacketTokenizer, decode_tokens_to_packets
)

PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"
PACKET_START = "<PACKET_START>"
PACKET_END = "<PACKET_END>"
STREAM_START = "<STREAM_START>"
STREAM_END = "<STREAM_END>"

def test_model(input_text, model_path="model_with_tokenizers.pth", device="cpu"):
    checkpoint = torch.load(model_path)
    text_tokenizer = TextTokenizer()
    text_tokenizer.word2id = checkpoint['text_tokenizer']
    text_tokenizer.id2word = {v: k for k, v in checkpoint['text_tokenizer'].items()}

    packet_tokenizer = PacketTokenizer()
    packet_tokenizer.token2id = checkpoint['packet_tokenizer']
    packet_tokenizer.id2token = {v: k for k, v in checkpoint['packet_tokenizer'].items()}

    model = Transformer(
        src_vocab_size=len(text_tokenizer.word2id),
        trg_vocab_size=len(packet_tokenizer.token2id),
        src_pad_idx=text_tokenizer.word2id[PAD],
        trg_pad_idx=packet_tokenizer.token2id[PAD],
        embed_size=512,
        num_layers=6,
        forward_expansion=8,
        heads=4,
        dropout=0.1,
        device=device,
        max_length=1024
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    generated_tokens = generate_packets(
        model, input_text, text_tokenizer, packet_tokenizer, device, max_len=100
    )

    print("\nGenerated Packet Tokens:")
    print(generated_tokens)

    decoded_packets = decode_tokens_to_packets(generated_tokens)
    print("\nDecoded Packets:")
    for idx, pkt in enumerate(decoded_packets):
        print(f"Packet {idx + 1}: {pkt}")


def main():
    mode = input("Enter 'train' to train the model or 'test' to test the model: ").strip().lower()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if mode == 'train':
        train_model()
    elif mode == 'test':
        user_prompt = input("Enter your text prompt for packet generation: ")
        test_model(user_prompt, device=device)
    else:
        print("Invalid mode. Please enter 'train' or 'test'.")


if __name__ == "__main__":
    main()
"""
Simulate the following stream: src_ip: 192.168.202.68, dst_ip: 192.168.24.100 and src_port: 8080 and dst_port: 1038
"""