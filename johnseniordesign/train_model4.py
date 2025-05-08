#!/usr/bin/env python3
 
import os
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformerModel2 import Transformer
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt

print("Using GPU:", torch.cuda.is_available())
overall_start_time = time.strftime("%H:%M:%S")
print("Start time:", overall_start_time)

from collections import Counter
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

def collate_fn(batch, text_tokenizer, packet_tokenizer):
    src_tensors = []
    trg_tensors = []
    for (prompt, pkt_tokens) in batch:
        src_ids = text_tokenizer.encode(prompt)
        pkt_ids = packet_tokenizer.encode(pkt_tokens)
        pkt_ids.append(packet_tokenizer.token2id[EOS])
        src_t = torch.tensor(src_ids, dtype=torch.long)
        trg_t = torch.tensor(pkt_ids, dtype=torch.long)
        src_tensors.append(src_t)
        trg_tensors.append(trg_t)
    src_padded = pad_sequence(
        src_tensors, batch_first=True, 
        padding_value=text_tokenizer.word2id[PAD]
    )
    trg_padded = pad_sequence(
        trg_tensors, batch_first=True, 
        padding_value=packet_tokenizer.token2id[PAD]
    )
    return src_padded, trg_padded

# --- Helper function to resize an embedding layer ---
def resize_embedding_layer(embedding_layer, new_vocab_size):
    old_weights = embedding_layer.weight.data
    old_vocab_size, embed_dim = old_weights.shape
    if new_vocab_size == old_vocab_size:
        return embedding_layer
    new_embedding = nn.Embedding(new_vocab_size, embed_dim)
    
    num_common = min(old_vocab_size, new_vocab_size)
    new_embedding.weight.data[:num_common] = old_weights[:num_common]
    return new_embedding


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device, text_tokenizer, packet_tokenizer):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    start_epoch = checkpoint.get('epoch', 0) + 1  # resume from next epoch

    saved_text_vocab_size = len(checkpoint['text_tokenizer'])
    saved_packet_vocab_size = len(checkpoint['packet_tokenizer'])
    current_text_vocab_size = text_tokenizer.vocab_size
    current_packet_vocab_size = packet_tokenizer.vocab_size

    if current_text_vocab_size != saved_text_vocab_size:
        print("Resizing source embedding layer from", saved_text_vocab_size, "to", current_text_vocab_size)
        model.encoder.embedding = resize_embedding_layer(model.encoder.embedding, current_text_vocab_size)

    if current_packet_vocab_size != saved_packet_vocab_size:
        print("Resizing target embedding layer from", saved_packet_vocab_size, "to", current_packet_vocab_size)
        model.decoder.embedding = resize_embedding_layer(model.decoder.embedding, current_packet_vocab_size)

    # Load the model state (using strict=False allows for differences)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    best_loss = checkpoint.get('best_loss', float('inf'))
    return start_epoch, best_loss


def train_transformer(
    model, 
    data_loader, 
    packet_tokenizer, 
    text_tokenizer, 
    optimizer,
    scheduler,
    start_epoch, 
    total_epochs, 
    device="cpu"
):
    criterion = nn.CrossEntropyLoss(
        ignore_index=packet_tokenizer.token2id[PAD], label_smoothing=0.1
    )
    
    def noam_scheduler(step, model_size, warmup_steps, factor=1.0):
        step = max(step, 1)
        return factor * ((model_size ** -0.5) *
                         min(step ** -0.5, step * (warmup_steps ** -1.5)))
    
    model.train()    
    best_loss = float('inf')
    losses = []
    
    for epoch in range(start_epoch, total_epochs):
        start_time = time.time()
        total_loss = 0.0
        for batch_idx, (src, trg) in enumerate(data_loader):
            src = src.to(device)
            trg = trg.to(device)
            trg_input = trg[:, :-1]
            trg_expected = trg[:, 1:].contiguous().view(-1)
            output = model(src, trg_input)
            output = output.view(-1, output.shape[2])
            loss = criterion(output, trg_expected)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        epoch_duration = time.time() - start_time
        avg_loss = total_loss / len(data_loader)
        minutes, seconds = divmod(epoch_duration, 60)
        print(f"Epoch [{epoch + 1}/{total_epochs}], Loss: {avg_loss:.4f}, Time: {int(minutes)}min {int(seconds)}sec")
        losses.append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'text_tokenizer': text_tokenizer.word2id,
                'packet_tokenizer': packet_tokenizer.token2id,
                'best_loss': best_loss
            }, 'model4.pth')
            print(f"Checkpoint saved at epoch {epoch + 1} with loss {avg_loss:.4f}")
            
    plt.plot(range(start_epoch + 1, total_epochs + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.show()

def top_k_logits(logits, k):
    values, indices = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1)
    neg_inf = torch.tensor(float('-inf'), dtype=logits.dtype, device=logits.device)
    return torch.where(logits < min_values, neg_inf, logits)

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
    JSON_PATH = "streams.json"
    dataset = MultiPacketStreamDataset(
        json_file=JSON_PATH,
        max_packets_per_stream=100
    )
    if len(dataset) == 0:
        print(f"No valid samples found in {JSON_PATH}.")
        return
    text_prompts = [sample[0] for sample in dataset]
    packet_data = [sample[1] for sample in dataset]
    text_tokenizer = TextTokenizer()
    text_tokenizer.build_vocab(text_prompts)
    packet_tokenizer = PacketTokenizer()
    flat_packet_data = [token for pkt_tokens in packet_data for token in pkt_tokens]
    packet_tokenizer.build_vocab([flat_packet_data])
    
    data_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, text_tokenizer, packet_tokenizer)
    )    
    
    src_vocab_size = text_tokenizer.vocab_size
    trg_vocab_size = packet_tokenizer.vocab_size
    src_pad_idx = text_tokenizer.word2id[PAD]
    trg_pad_idx = packet_tokenizer.token2id[PAD]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        embed_size=512,
        num_layers=6,
        forward_expansion=8,
        heads=4,
        dropout=0.1,
        device=device,
        max_length=1024
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {param_count}")
    
    # Initialize optimizer and scheduler outside the training loop.
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    def noam_scheduler(step):
        return ((512 ** -0.5) * min(max(step,1) ** -0.5, max(step,1) * (4000 ** -1.5)))
    scheduler = LambdaLR(optimizer, lr_lambda=noam_scheduler)
    
    total_epochs = 30
    start_epoch = 0
    best_loss = float('inf')
    checkpoint_path = 'model4.pth'
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint to resume training...")
        start_epoch, best_loss = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device, text_tokenizer, packet_tokenizer)
    
    train_transformer(
        model=model,
        data_loader=data_loader,
        packet_tokenizer=packet_tokenizer,
        text_tokenizer=text_tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        total_epochs=total_epochs,
        device=device
    )
    
    user_prompt = ("Simulate a multi-packet (protocol) TCP exchange between 192.168.10.5 (port 50000) "
                   "and 192.168.10.7 (port 80). Start with a three-way handshake using the flags SYN, SYN/ACK, and ACK. "
                   "Then send a packet with 120 bytes of payload.")
    generated_tokens = generate_packets(model, user_prompt, text_tokenizer, packet_tokenizer, device, max_len=100)
    print("\nGenerated Packet Tokens:")
    print(generated_tokens)
    
    decoded_packets = decode_tokens_to_packets(generated_tokens)
    overall_end_time = time.strftime("%H:%M:%S")
    start_dt = datetime.strptime(overall_start_time, "%H:%M:%S")
    end_dt = datetime.strptime(overall_end_time, "%H:%M:%S")
    overall_run_time = end_dt - start_dt
    print("End time: ", overall_end_time)
    print("Total runtime: ", overall_run_time)
    print("\nDecoded Packets:")
    for idx, pkt in enumerate(decoded_packets):
        print(f"Packet {idx + 1}: {pkt}")

if __name__ == "__main__":
    main()
