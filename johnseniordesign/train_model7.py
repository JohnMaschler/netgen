#!/usr/bin/env python3

import os
import json
import sqlite3
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from transformerModel2 import Transformer
from torch.optim.lr_scheduler import LambdaLR

from pcap_tokenization3 import (
    TextTokenizer, PacketTokenizer, decode_tokens_to_packets
)

from spm import SentencePieceTokenizer

PAD = "<PAD>"
SOS = "<SOS>"
EOS = "<EOS>"
PACKET_START = "<PACKET_START>"
PACKET_END = "<PACKET_END>"
STREAM_START = "<STREAM_START>"
STREAM_END = "<STREAM_END>"

def collate_fn(batch, text_tokenizer, packet_tokenizer):
    src_tensors = []
    trg_tensors = []

    for (prompt, pkt_tokens) in batch:
        # Encode the prompt and tokens
        src_ids = text_tokenizer.encode(prompt)
        pkt_ids = packet_tokenizer.encode(pkt_tokens)
        pkt_ids.append(packet_tokenizer.token2id[EOS]) 

        # Convert to tensors for padding
        src_t = torch.tensor(src_ids, dtype=torch.long)
        trg_t = torch.tensor(pkt_ids, dtype=torch.long)

        src_tensors.append(src_t)
        trg_tensors.append(trg_t)

    # Pad sequences 
    src_padded = pad_sequence(
        src_tensors, batch_first=True, 
        padding_value=text_tokenizer.pad_id
    )
    trg_padded = pad_sequence(
        trg_tensors, batch_first=True, 
        padding_value=packet_tokenizer.token2id[PAD]
    )

    return src_padded, trg_padded

class MultiPacketStreamDataset(Dataset):
    """
    A dataset that loads packet streams from a local SQLite DB instead of a JSON file.
    Each sample is a tuple: (prompt, packets)
    """
    def __init__(self, db_path, max_packets_per_stream=100):
        self.db_path = db_path
        self.max_packets_per_stream = max_packets_per_stream
        self.streams = self.load_streams_from_db()

    def load_streams_from_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT description, prompt, tags, src_ip, dst_ip, src_port, dst_port, protocol, packets FROM streams")
        rows = c.fetchall()
        streams = []
        for row in rows:
            description, prompt, tags, src_ip, dst_ip, src_port, dst_port, protocol, packets_str = row
            packets = json.loads(packets_str)
            if len(packets) >= self.max_packets_per_stream:
                streams.append((prompt, packets[:self.max_packets_per_stream]))
        conn.close()
        return streams

    def __len__(self):
        return len(self.streams)

    def __getitem__(self, idx):
        return self.streams[idx]

def train_transformer(
    model, 
    data_loader, 
    packet_tokenizer, 
    text_tokenizer, 
    epochs=500,  # Increased to 500 epochs
    lr=1e-3, 
    device="cpu",
    snapshot_interval=10,  # Save snapshot every 10 epochs
    convergence_threshold=1e-5,  # Stop training if loss reaches this value
    checkpoint_dir="checkpoints"  # Directory to save snapshots
):
    import os
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    criterion = nn.NLLLoss(ignore_index=packet_tokenizer.token2id[PAD])
    
    def noam_scheduler(step, model_size, warmup_steps, factor=1.0):
        if step == 0:
            step = 1
        return factor * (
            (model_size ** -0.5) *
            min(step ** -0.5, step * (warmup_steps ** -1.5))
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = LambdaLR(optimizer, 
                lr_lambda=lambda step: noam_scheduler(step, 512, 4000, factor=1.0))

    model.train()
    
    best_loss = float('inf')
    losses = []
    last_improvement = 0
    # patience = 20  # Stop if no improvement for this many epochs

    # Check if a checkpoint exists to resume from
    latest_checkpoint = None
    start_epoch = 0

    for epoch in range(start_epoch, epochs):
        total_loss = 0.0
        for batch_idx, (src, trg) in enumerate(data_loader):
            src = src.to(device)
            trg = trg.to(device)

            trg_input = trg[:, :-1]
            trg_expected = trg[:, 1:].contiguous().view(-1)

            output = model(src, trg_input)
            output = output.view(-1, output.shape[2])
            log_probs = torch.log_softmax(output, dim=1)
            loss = criterion(log_probs, trg_expected)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}")
        losses.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            last_improvement = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'best_loss': best_loss,
                'losses': losses,
                'text_tokenizer': "my_spm.model",
                'packet_tokenizer': packet_tokenizer.token2id
            }, 'best_model6.pth')
            print(f"Best model saved with loss {avg_loss:.6f}")
        
        if avg_loss <= convergence_threshold:
            print(f"Convergence achieved at epoch {epoch+1} with loss {avg_loss:.6f}")
            break

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()
    
    return model

def top_k_logits(logits, k):
    values, indices = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1)
    return torch.where(logits < min_values, float('-inf'), logits)

def generate_packets(model, input_text, text_tokenizer, packet_tokenizer, device, max_len=100, k=5):
    model.eval()
    with torch.no_grad():
        src = text_tokenizer.encode(input_text)
        src_tensor = torch.tensor([src], dtype=torch.long).to(device)

        trg_indices = [packet_tokenizer.token2id[STREAM_START]]
        token_counts = Counter()

        for _ in range(max_len):
            trg_tensor = torch.tensor([trg_indices], dtype=torch.long).to(device)
            output = model(src_tensor, trg_tensor)
            logits = output[:, -1, :]
            
            top_k_filtered_logits = top_k_logits(logits, k)
            probabilities = torch.softmax(top_k_filtered_logits, dim=1)
            next_token = torch.multinomial(probabilities, 1).item()
            
            if next_token == packet_tokenizer.token2id[EOS]:
                break

            trg_indices.append(next_token)
            token_counts[next_token] += 1

        trg_indices.append(packet_tokenizer.token2id[STREAM_END])
        return packet_tokenizer.decode(trg_indices[1:-1])

def main():
    DB_PATH = "streams.db"
    dataset = MultiPacketStreamDataset(
        db_path=DB_PATH,
        max_packets_per_stream=100
    )

    if len(dataset) == 0:
        print(f"No valid samples found in the database at {DB_PATH}.")
        return

    text_prompts = [sample[0] for sample in dataset]
    packet_data = [sample[1] for sample in dataset]

    text_tokenizer = SentencePieceTokenizer(model_path="my_spm.model")
    print("spm vocab size: ", text_tokenizer.vocab_size)

    packet_tokenizer = PacketTokenizer()

    flat_packet_data = []
    for pkt_tokens in packet_data:
        for token in pkt_tokens:
            flat_packet_data.append(token)
    packet_tokenizer.build_vocab([flat_packet_data])

    data_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, text_tokenizer, packet_tokenizer)
    )    

    src_vocab_size = text_tokenizer.vocab_size
    trg_vocab_size = packet_tokenizer.vocab_size
    src_pad_idx = text_tokenizer.pad_id
    trg_pad_idx = packet_tokenizer.token2id[PAD]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        embed_size=512,
        num_layers=6,
        forward_expansion=6,
        heads=8,
        dropout=0.1,
        device=device,
        max_length=1024
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {param_count}")

    train_transformer(
        model=model,
        data_loader=data_loader,
        packet_tokenizer=packet_tokenizer,
        text_tokenizer=text_tokenizer,
        epochs=1000,
        lr=1e-3,
        device=device,
        snapshot_interval=10,
        convergence_threshold=1e-5,
        checkpoint_dir="model_checkpoints"
    )

    user_prompt = ("Simulate a multi-packet (protocol) TCP exchange between 192.168.10.5 (port 50000) "
                   "and 192.168.10.7 (port 80). Start with a three-way handshake using the flags SYN, SYN/ACK, and ACK. "
                   "Then send a packet with 120 bytes of payload.")
    generated_tokens = generate_packets(model, user_prompt, text_tokenizer, packet_tokenizer, device, max_len=100)
    print("\nGenerated Packet Tokens:")
    print(generated_tokens)

    decoded_packets = decode_tokens_to_packets(generated_tokens)
    print("\nDecoded Packets:")
    for idx, pkt in enumerate(decoded_packets):
        print(f"Packet {idx + 1}: {pkt}")

if __name__ == "__main__":
    main()
