#!/usr/bin/env python3
"""
Test a trained NetGen Transformer checkpoint (zip or legacy-pickle).

Example:
    python test_model5.py --checkpoint best_model7_db_2.pth \
        --prompt "Generate TCP traffic between 10.0.0.1 and 10.0.0.2"
"""

import argparse
import os
import pickle
import torch
import sentencepiece as spm
import sys
import glob
import random


from transformerModel import Transformer
from pcap_tokenization_db import PacketTokenizer, decode_tokens_to_packets


def find_checkpoint_files(directory='.'):
    """Find all .pth and .pt files in the directory"""
    checkpoint_files = []
    for ext in ['*.pth', '*.pt', '*.ckpt']:
        checkpoint_files.extend(glob.glob(os.path.join(directory, ext)))
    return checkpoint_files

def load_checkpoint_with_recovery(path, map_location="cpu"):
    """
    Enhanced checkpoint loader with better error recovery
    """
    print(f"Attempting to load checkpoint: {path}")
    
    try:
        print("Trying torch.load with weights_only=True...")
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e:
        print(f"Failed with weights_only=True: {e}")
    
    try:
        print("Trying torch.load with default settings...")
        return torch.load(path, map_location=map_location)
    except Exception as e:
        print(f"Failed with default settings: {e}")
    
    try:
        print("Trying pickle with latin1 encoding...")
        with open(path, 'rb') as f:
            return pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f"Failed with latin1 encoding: {e}")
    
    try:
        print("Trying pickle in binary mode...")
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Failed with binary mode: {e}")
        
    
    checkpoint_files = find_checkpoint_files()
    if checkpoint_files:
        print("\nAvailable checkpoint files:")
        for i, file in enumerate(checkpoint_files):
            print(f"  {i+1}. {file}")
        
    raise RuntimeError(f"Could not load checkpoint {path}")


class SentencePieceTokenizer:
    def __init__(self, model_path="my_spm_new2.model"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SentencePiece model not found: {model_path}")
            
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

        self.pad_id = self.sp.PieceToId("<PAD>")
        self.unk_id = self.sp.PieceToId("<UNK>")
        self.bos_id = self.sp.PieceToId("< SOS >")
        self.eos_id = self.sp.PieceToId("<EOS>")
        self.vocab_size = self.sp.GetPieceSize()

    def encode(self, text: str):
        return [self.bos_id] + self.sp.EncodeAsIds(text) + [self.eos_id]

    def decode(self, ids):
        keep = [i for i in ids if i not in (self.bos_id, self.eos_id, self.pad_id)]
        return self.sp.DecodeIds(keep)


def create_mock_model_checkpoint(force_create=False):
    """Create a simple mock checkpoint for testing when real checkpoint isn't available"""
    if os.path.exists('mock_model.pth') and not force_create:
        return 'mock_model.pth'
        
    print("Creating mock model checkpoint for testing...")
    mock_model = Transformer(
        src_vocab_size=10000,
        trg_vocab_size=10000,
        src_pad_idx=0,
        trg_pad_idx=0,
        embed_size=512,
        num_layers=6,
        forward_expansion=6,
        heads=8,
        dropout=0.1,
        device="cpu",
        max_length=1024,
    )
    
    mock_tokenizer = PacketTokenizer()
    mock_tokenizer.token2id.update({
        f"token_{i}": i for i in range(100, 10000)
    })
    
    mock_checkpoint = {
        "model_state_dict": mock_model.state_dict(),
        "packet_tokenizer": mock_tokenizer.token2id,
        "text_tokenizer": "my_spm_new2.model"
    }
    
    torch.save(mock_checkpoint, 'mock_model.pth')
    print("mock checkpoint: mock_model.pth")
    return 'mock_model.pth'


def find_sentencepiece_models(directory='.'):
    """Find all .model files in the directory"""
    return glob.glob(os.path.join(directory, '*.model'))


def load_tokenizers_from_checkpoint(ckpt):
    sp_path = ckpt.get("text_tokenizer", "my_spm_new2.model")
    model_files = find_sentencepiece_models()
    
    if not os.path.exists(sp_path) and model_files:
        print(f"SentencePiece model not found at {sp_path}")
        print(f"model files: {model_files}")
        sp_path = model_files[0]
    
    if not os.path.exists(sp_path):
        print(f"No SentencePiece model found")
        print("Creating a simple mock SentencePiece model for testing...")
        try:
            import tempfile
            vocab_size = 10000
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
                for i in range(vocab_size):
                    f.write(f"token{i}\n".encode())
            
            tmp_model_prefix = "temp_spm"
            spm.SentencePieceTrainer.Train(
                f'--input={f.name} --model_prefix={tmp_model_prefix} '
                f'--vocab_size={vocab_size} --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 '
                '--character_coverage=1.0 --model_type=unigram'
            )
            os.unlink(f.name)
            sp_path = f"{tmp_model_prefix}.model"
            print(f"Created temporary model: {sp_path}")
        except Exception as e:
            print(f"Failed to create mock model: {e}")
    
    try:
        txt_tok = SentencePieceTokenizer(sp_path)
        print(f"Loaded SentencePiece model: {sp_path}")
    except Exception as e:
        print(f"Error loading SentencePiece model: {e}")
        raise
    
    pkt_tok = PacketTokenizer()
    if "packet_tokenizer" in ckpt:
        pkt_tok.token2id = ckpt["packet_tokenizer"]
    else:
        print("No packet tokenizer found in checkpoint, using default")
    
    pkt_tok.id2token = {v: k for k, v in pkt_tok.token2id.items()}
    pkt_tok.vocab_size = len(pkt_tok.token2id)
    return txt_tok, pkt_tok

def generate_packets(model, prompt, txt_tok, pkt_tok, device, *, max_len=100, k=5, 
                    temp=1.0, p=0.9, use_sampling="topk"):
    """
    Generate packets using different sampling strats for testing:
    - topk: Sample from top-k highest probability tokens
    - topp: Nucleus sampling (sample from smallest set of tokens with cumulative probability >= p)
    - temp: Apply temperature to soften/sharpen the distribution
    - greedy: Always select the highest probability token (deterministic)
    - beam: Simple beam search with beam_size=k
    """
    model.eval()
    with torch.no_grad():
        src = torch.tensor([txt_tok.encode(prompt)], dtype=torch.long, device=device)

        trg_ids = []
        if ("<STREAM_START>" in pkt_tok.token2id):
            trg_ids.append(pkt_tok.token2id["<STREAM_START>"])
            
        # beam search
        if (use_sampling == "beam"):
            beam_size = min(k, 10)  # Limit beam size
            beams = [(trg_ids, 0)]  # (sequence, score)
            
            for _ in range(max_len):
                new_beams = []
                
                for beam_seq, beam_score in beams:
                    if (beam_seq and beam_seq[-1] == pkt_tok.token2id.get("<STREAM_END>", -1)):
                        # Keep completed sequences
                        new_beams.append((beam_seq, beam_score))
                        continue
                        
                    trg = torch.tensor([beam_seq], dtype=torch.long, device=device)
                    try:
                        logits = model(src, trg)[:, -1, :]
                        log_probs = torch.log_softmax(logits, dim=1)
                        
                        # Get top k options
                        topk_probs, topk_ids = torch.topk(log_probs, beam_size)
                        
                        for i in range(beam_size):
                            token_id = topk_ids[0, i].item()
                            token_score = topk_probs[0, i].item()
                            new_seq = beam_seq + [token_id]
                            new_score = beam_score + token_score
                            new_beams.append((new_seq, new_score))
                    except Exception as e:
                        print(f"Error during beam search: {e}")
                        continue
                
                # Truncate and keep top beams
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
                
                # If all beams are complete, break
                if (all(seq[-1] == pkt_tok.token2id.get("<STREAM_END>", -1) for seq, _ in beams)):
                    break
            
            # Take the best beam
            trg_ids = beams[0][0]
            
        else:
            # auto-regressive generation
            for _ in range(max_len):
                trg = torch.tensor([trg_ids], dtype=torch.long, device=device)
                
                try:
                    logits = model(src, trg)[:, -1, :]  # (1, vocab)
                    
                    if (temp != 1.0):
                        logits = logits / temp
                    
                    if (use_sampling == "greedy"):
                        # Greedy (deterministic)
                        next_id = torch.argmax(logits, dim=1).item()
                    
                    elif (use_sampling == "topp"):
                        # Top-p (nucleus) sampling
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=1), dim=1)
                        
                        sorted_indices_to_remove = cumulative_probs > p
                        # Shift the indices to the right to keep the first token above the threshold
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        logits[indices_to_remove] = float('-inf')
                        
                        probs = torch.softmax(logits, dim=1)
                        next_id = torch.multinomial(probs, 1).item()
                    
                    else:  # "topk" (default)
                        topk_vals = torch.topk(logits, min(k, logits.size(-1)))
                        thresh = topk_vals.values[:, -1].unsqueeze(1)
                        logits = torch.where(logits < thresh, float("-inf"), logits)
                        probs = torch.softmax(logits, dim=1)
                        next_id = torch.multinomial(probs, 1).item()
                
                except Exception as e:
                    print(f"Error during generation: {e}")
                    break

                if ("<EOS>" in pkt_tok.token2id and next_id == pkt_tok.token2id["<EOS>"]):
                    break
                    
                # Stop on STREAM_END
                # if ("<STREAM_END>" in pkt_tok.token2id and next_id == pkt_tok.token2id["<STREAM_END>"]):
                #     break

                trg_ids.append(next_id)

        # Only add stream_end if it wasn't already generated
        if ("<STREAM_END>" in pkt_tok.token2id and not (trg_ids and trg_ids[-1] == pkt_tok.token2id["<STREAM_END>"])):
            trg_ids.append(pkt_tok.token2id["<STREAM_END>"])

        return pkt_tok.decode(trg_ids)


def main():
    ap = argparse.ArgumentParser("NetGen checkpoint tester")
    ap.add_argument("--checkpoint", default="best_model7_db_3.pth", 
                    help="path to .pth checkpoint")
    ap.add_argument("--prompt", required=True,
                    help="natural-language prompt for generation")
    ap.add_argument("--max-length", type=int, default=100,
                    help="maximum sequence length to generate")
    ap.add_argument("--sampling", choices=["topk", "topp", "greedy", "beam", "temp"], default="topk",
                    help="sampling strategy to use: topk, topp (nucleus), greedy, beam, or temperature")
    ap.add_argument("--topk", type=int, default=5,
                    help="number of tokens to consider in top-k sampling or beam search")
    ap.add_argument("--topp", type=float, default=0.9, 
                    help="probability threshold for nucleus sampling")
    ap.add_argument("--temp", type=float, default=1.0,
                    help="temperature for softening/sharpening distribution")
    ap.add_argument("--mock", action="store_true",
                    help="use mock model if checkpoint loading fails")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Try to load the checkpoint
    try:
        if not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(f"❌ {args.checkpoint} does not exist")
        
        ckpt = load_checkpoint_with_recovery(args.checkpoint, map_location=device)
        print("Checkpoint loaded successfully.")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        
        if args.mock:
            print("\nUsing mock model for testing purposes")
            mock_path = create_mock_model_checkpoint()
            ckpt = load_checkpoint_with_recovery(mock_path, map_location=device)
        else:
            available_ckpts = find_checkpoint_files()
            if available_ckpts:
                print("\nAvailable checkpoint files:")
                for i, file in enumerate(available_ckpts):
                    print(f"  {i+1}. {file}")
                print("\nTry with --checkpoint or use --mock to test with a very small mock model")
            sys.exit(1)

    # if "model_state_dict" not in ckpt:
    #     print("model_state_dict' missing from checkpoint, alternatives:")
        
    #     state_dict_keys = [k for k in ckpt.keys() if isinstance(ckpt[k], dict) and 
    #                        any('weight' in str(key) for key in ckpt[k].keys())]
    #     if state_dict_keys:
    #         print(f"Using '{state_dict_keys[0]}' as model state dict")
    #         ckpt["model_state_dict"] = ckpt[state_dict_keys[0]]
    #     else:
    #         print("No state dict found in checkpoint")
    #         if args.mock:
    #             print("Creating mock state dict")
    #             mock_path = create_mock_model_checkpoint(force_create=True)
    #             mock_ckpt = torch.load(mock_path)
    #             ckpt["model_state_dict"] = mock_ckpt["model_state_dict"]
    #         else:
    #             raise KeyError("Could not find model state dictionary in checkpoint")

    # Load tokenizers
    txt_tok, pkt_tok = load_tokenizers_from_checkpoint(ckpt)

    # Initialize model
    model = Transformer(
        src_vocab_size=txt_tok.vocab_size,
        trg_vocab_size=pkt_tok.vocab_size,
        src_pad_idx=txt_tok.pad_id,
        trg_pad_idx=pkt_tok.token2id["<PAD>"] if "<PAD>" in pkt_tok.token2id else 0,
        embed_size=512,
        num_layers=6,
        forward_expansion=6,
        heads=8,
        dropout=0.1,
        device=device,
        max_length=1024,
    ).to(device)

    try:
        model.load_state_dict(ckpt["model_state_dict"])
        print("Model weights restored.")
        
        # Count and print total parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nModel Statistics:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        
        # Print parameters by layer type
        encoder_params = sum(p.numel() for name, p in model.named_parameters() if 'encoder' in name)
        decoder_params = sum(p.numel() for name, p in model.named_parameters() if 'decoder' in name)
        embedding_params = sum(p.numel() for name, p in model.named_parameters() if 'embedding' in name)
        
        print(f"\nParameters by component:")
        print(f"Encoder: {encoder_params:,}")
        print(f"Decoder: {decoder_params:,}")
        print(f"Embeddings: {embedding_params:,}")
        
    except Exception as e:
        print(f"Error loading model weights: {e}")
        
        if args.mock:
            print("\nUsing mock model weights")
            mock_path = create_mock_model_checkpoint(force_create=True)
            mock_ckpt = torch.load(mock_path)
            model = Transformer(
                src_vocab_size=txt_tok.vocab_size,
                trg_vocab_size=pkt_tok.vocab_size,
                src_pad_idx=txt_tok.pad_id,
                trg_pad_idx=pkt_tok.token2id["<PAD>"] if "<PAD>" in pkt_tok.token2id else 0,
                embed_size=512,
                num_layers=6,
                forward_expansion=6,
                heads=8,
                dropout=0.1,
                device=device,
                max_length=1024,
            ).to(device)
        else:
            print("\nTry running with --mock to test with a mock model")
            sys.exit(1)

    print(f"\nGenerating packets for prompt: '{args.prompt}'")
    try:
        tokens = generate_packets(
            model, args.prompt, txt_tok, pkt_tok, device, 
            max_len=args.max_length, 
            k=args.topk,
            temp=args.temp,
            p=args.topp,
            use_sampling=args.sampling
        )
        print("\nGenerated tokens:")
        print(tokens)

        # mock model, some sample packets for demonstration
        if args.mock:
            print("\nDecoded packets (mock data):")
            
            prompt_parts = args.prompt.split()
            src_ip = "192.168.1.10"
            dst_ip = "8.8.8.8"
            protocol = "TCP"
            dst_port = "53"
            
            # Extract values from prompt if present
            for i, part in enumerate(prompt_parts):
                if "src_ip:" in part:
                    src_ip = part.split(":")[1].strip()
                elif "dst_ip:" in part:
                    dst_ip = part.split(":")[1].strip()
                elif "protocol:" in part:
                    protocol = part.split(":")[1].strip()
                elif "dst_port:" in part:
                    dst_port = part.split(":")[1].strip()
            
            # Generate random source port for demonstration
            src_port = str(random.randint(10000, 65000))
            
            sample_packets = [
                {
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "protocol": protocol,
                    "src_port": src_port,
                    "dst_port": dst_port,
                    "timestamp": "1.000000",
                    "flags": "S",
                    "payload_size": "0"
                },
                {
                    "src_ip": dst_ip,
                    "dst_ip": src_ip,
                    "protocol": protocol,
                    "src_port": dst_port,
                    "dst_port": src_port,
                    "timestamp": "1.000500",
                    "flags": "SA",
                    "payload_size": "0"
                },
                {
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "protocol": protocol,
                    "src_port": src_port,
                    "dst_port": dst_port,
                    "timestamp": "1.001000",
                    "flags": "A",
                    "payload_size": "0" 
                }
            ]
            
            # Display the sample packets
            for i, pkt in enumerate(sample_packets, 1):
                print(f"Packet {i}: {pkt}")
        else:
            try:
                packets = decode_tokens_to_packets(tokens)
                print("\nDecoded packets:")
                for i, pkt in enumerate(packets, 1):
                    print(f"Packet {i}: {pkt}")
            except Exception as e:
                print(f"\nError decoding packets: {e}")
                print("Using token information directly:")
                
                # Try to extract structured data from tokens
                special_tokens = ["<PACKET_START>", "<PACKET_END>", "<PAYLOAD_START>", 
                                  "<PAYLOAD_END>", "<STREAM_START>", "<STREAM_END>"]
                current_packet = {}
                packets = []
                
                for token in tokens:
                    if (token == "<PACKET_START>"):
                        current_packet = {}
                    elif (token == "<PACKET_END>"):
                        if current_packet:
                            packets.append(current_packet)
                    elif (token not in special_tokens):
                        # Try to parse token as field:value
                        parts = token.split(":", 1)
                        if (len(parts) == 2):
                            field, value = parts
                            current_packet[field] = value
                
                if packets:
                    for i, pkt in enumerate(packets, 1):
                        print(f"Packet {i}: {pkt}")
                else:
                    print("error")
                    
    except Exception as e:
        print(f"Error during generation: {e}")


if __name__ == "__main__":
    main()
    
    
