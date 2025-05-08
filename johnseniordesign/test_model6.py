#!/usr/bin/env python3
"""
Supports two modes:
  1. single: emit one full stream
  2. flow: repeatedly sample streams and accumulate ≥ --min-packets unique packets

Example (single):
    python test_model6.py \
      --checkpoint ckpt.pth \
      --prompt "Generate TCP traffic between 10.0.0.1 and 10.0.0.2" \
      --mode single

Example (flow):
    python test_model6.py \
      --checkpoint ckpt.pth \
      --prompt "simulate TCP connection ..." \
      --mode flow --min-packets 15 --max-tries 30 \
      --sampling topp --topp 0.9 --temp 1.3 \
      --skip-steps 12 --rep-penalty 0.85 \
      --end-penalty 0.4 --pktend-bonus 1.2
"""

import argparse, os, pickle, glob, random
import torch, sentencepiece as spm
from transformerModel import Transformer
from pcap_tokenization_db import PacketTokenizer, decode_tokens_to_packets

def find_checkpoint_files(directory='.'):
    files = []
    for ext in ('*.pth','*.pt','*.ckpt'):
        files += glob.glob(os.path.join(directory, ext))
    return files

def load_checkpoint_with_recovery(path, map_location='cpu'):
    print(f"Attempting to load checkpoint: {path}")
    try:
        print("Trying torch.load with default settings")
        return torch.load(path, map_location=map_location)
    except Exception as e:
        print(f"Failed: {e}")
    
    # Fallback to pickle if torch.load fails
    for enc in ('latin1', None):
        try:
            print(f"Trying pickle, encoding={enc}")
            with open(path, 'rb') as f:
                return pickle.load(f, encoding=enc) if enc else pickle.load(f)
        except Exception as e:
            print(f"Failed: {e}")
    
    # If we've reached here, show available checkpoints and fail
    checkpoint_files = find_checkpoint_files(os.path.dirname(path))
    if checkpoint_files:
        print("\nAvailable checkpoint files:")
        for i, file in enumerate(checkpoint_files, 1):
            print(f"  {i}. {file}")
            
    raise RuntimeError(f"Could not load checkpoint {path}")

class SentencePieceTokenizer:
    def __init__(self, model_path="my_spm_new2.model"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        self.sp = spm.SentencePieceProcessor(); self.sp.Load(model_path)
        self.pad_id = self.sp.PieceToId("<PAD>")
        self.bos_id = self.sp.PieceToId("<SOS>")
        self.eos_id = self.sp.PieceToId("<EOS>")
        self.vocab_size = self.sp.GetPieceSize()
    def encode(self, text):
        return [self.bos_id] + self.sp.EncodeAsIds(text) + [self.eos_id]
    def decode(self, ids):
        skip = {self.pad_id, self.bos_id, self.eos_id}
        return self.sp.DecodeIds([i for i in ids if i not in skip])

def generate_packets(
    model, prompt, txt_tok, pkt_tok, device,
    max_len=200, k=5, temp=1.0, p=0.9, use_sampling='topk',
    skip_steps=15, rep_penalty=0.9, end_penalty=0.5, pktend_bonus=1.2
):
    """
    Generate one full stream, with repetition-, length- and boundary-penalties.
    """
    model.eval()
    STREAM = pkt_tok.token2id['<STREAM_START>']
    END    = pkt_tok.token2id.get('<STREAM_END>')
    PKTEND = pkt_tok.token2id.get('<PACKET_END>')
    out = [STREAM]
    src = torch.tensor([txt_tok.encode(prompt)], device=device)
    with torch.no_grad():
        for step in range(max_len):
            trg = torch.tensor([out], device=device)
            logits = model(src, trg)[:, -1, :].clone()
            # temperature
            if temp != 1.0:
                logits /= temp
            # mask + penalize early END
            if END is not None:
                if step < skip_steps:
                    logits[:, END] = -1e9
                logits[:, END] *= end_penalty
            # boost packet boundaries
            if PKTEND is not None:
                logits[:, PKTEND] *= pktend_bonus
            # repetition penalty
            for t in set(out):
                logits[:, t] *= rep_penalty
            # sampling
            if use_sampling in ('greedy','beam'):
                nid = int(torch.argmax(logits, dim=1))
            else:
                if use_sampling == 'topp':
                    vals, idx = torch.sort(logits, descending=True)
                    probs = torch.softmax(vals, 1)
                    cum = torch.cumsum(probs, 1)
                    mask = cum < p; mask[...,0] = True
                    full = torch.zeros_like(logits, dtype=torch.bool)
                    full.scatter_(1, idx, mask)
                    logits = logits.masked_fill(~full, float('-inf'))
                else:  # topk
                    topv, topi = torch.topk(logits, min(k, logits.size(-1)))
                    thresh = topv[:, -1].unsqueeze(1)
                    logits = torch.where(logits < thresh, float('-inf'), logits)
                nid = int(torch.multinomial(torch.softmax(logits,1), 1))
            out.append(nid)
            if nid == END:
                break
    return pkt_tok.decode(out)

def standardize_packet_format(packet):
    """Standardize packet fields in a consistent order."""
    standard_order = [
        'src_ip', 'dst_ip', 'protocol', 'src_port', 'dst_port',
        'flags', 'timestamp', 'payload_size'
    ]
    
    ordered_packet = {}
    for field in standard_order:
        if field in packet:
            ordered_packet[field] = packet[field]
    
    # Add any remaining fields that weren't in our standard order
    for field, value in packet.items():
        if field not in ordered_packet:
            ordered_packet[field] = value
            
    return ordered_packet

def build_flow(
    model, prompt, txt_tok, pkt_tok, device,
    min_packets=10, max_tries=20, **gen_kwargs
):
    """
    Keep sampling full streams until we collect
    at least min_packets _unique_ packets (or exhaust max_tries).
    """
    flow, seen, tries = [], set(), 0
    while len(flow) < min_packets and tries < max_tries:
        tries += 1
        seq = generate_packets(model, prompt, txt_tok, pkt_tok, device, **gen_kwargs)
        try:
            pkts = decode_tokens_to_packets(seq)
            # Standardize the packet format
            pkts = [standardize_packet_format(p) for p in pkts]
        except Exception:
            pkts = []
        new = []
        for p in pkts:
            key = repr(p)
            if key not in seen:
                seen.add(key)
                new.append(p)
        if new:
            flow.extend(new)
            print(f"  try {tries}: added {len(new)} packets → total {len(flow)}")
        else:
            print(f"  try {tries}: no new packets")
    return flow

def main():
    p = argparse.ArgumentParser("NetGen tester")
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--prompt',    required=True)
    p.add_argument('--mode',      choices=['single','flow'], default='single')
    p.add_argument('--max-length',type=int,   default=200)
    p.add_argument('--sampling',  choices=['topk','topp','greedy','beam'], default='topp')
    p.add_argument('--topk',      type=int,   default=8)
    p.add_argument('--topp',      type=float, default=0.92)
    p.add_argument('--temp',      type=float, default=1.3)
    p.add_argument('--min-packets',type=int,  default=15)
    p.add_argument('--max-tries', type=int,   default=30)
    p.add_argument('--skip-steps',type=int,   default=15,
                   help='mask <STREAM_END> for this many tokens')
    p.add_argument('--rep-penalty',type=float,default=0.9,
                   help='penalize repeats (0–1)')
    p.add_argument('--end-penalty',type=float,default=0.5,
                   help='penalize early end (0–1)')
    p.add_argument('--pktend-bonus',type=float,default=1.2,
                   help='boost packet boundaries (>1)')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = load_checkpoint_with_recovery(args.checkpoint, map_location=device)

    # load tokenizers
    txt_tok = SentencePieceTokenizer(ckpt.get('text_tokenizer','my_spm_new2.model'))
    pkt_tok = PacketTokenizer()
    pkt_tok.token2id = ckpt.get('packet_tokenizer', pkt_tok.token2id)
    pkt_tok.id2token = {v:k for k,v in pkt_tok.token2id.items()}

    # unwrap state_dict whether we loaded full ckpt or just state_dict:
    state_dict = (
        ckpt['model_state_dict']
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt
        else ckpt
    )

    for name, weight in state_dict.items():
        if name.startswith('decoder.') and name.endswith('word_embedding.weight'):
            trg_vocab_size = weight.shape[0]
            print(f"Detected decoder vocab size: {trg_vocab_size}")
            pkt_tok.vocab_size = trg_vocab_size
            break
    else:
        # fallback
        pkt_tok.vocab_size = len(pkt_tok.token2id)
        print(f"No decoder.embedding found, using packet_tokenizer size {pkt_tok.vocab_size}")

    # build model with correct vocab sizes
    model = Transformer(
        src_vocab_size=txt_tok.vocab_size,
        trg_vocab_size=pkt_tok.vocab_size,
        src_pad_idx=txt_tok.pad_id,
        trg_pad_idx=pkt_tok.token2id.get('<PAD>',0),
        embed_size=512, num_layers=6, forward_expansion=6,
        heads=8, dropout=0.1, device=device, max_length=1024
    ).to(device)
    model.load_state_dict(state_dict)
    print("✅ Loaded checkpoint")

    if args.mode == 'single':
        seq = generate_packets(
            model, args.prompt, txt_tok, pkt_tok, device,
            max_len=args.max_length,
            k=args.topk, temp=args.temp, p=args.topp,
            use_sampling=args.sampling,
            skip_steps=args.skip_steps,
            rep_penalty=args.rep_penalty,
            end_penalty=args.end_penalty,
            pktend_bonus=args.pktend_bonus
        )
        print("\n" + seq)
        
        # Add this to standardize the packet display for single mode too
        try:
            pkts = decode_tokens_to_packets(seq)
            pkts = [standardize_packet_format(p) for p in pkts]
            print("\nStandardized packets:")
            for i, p in enumerate(pkts, 1):
                print(f"Packet {i}: {p}")
        except Exception as e:
            print(f"Error decoding packets: {e}")
    else:
        print(f"\nBuilding flow of ≥ {args.min_packets} packets…")
        flow = build_flow(
            model, args.prompt, txt_tok, pkt_tok, device,
            min_packets=args.min_packets,
            max_tries=args.max_tries,
            max_len=args.max_length,
            k=args.topk,
            temp=args.temp,
            p=args.topp,
            use_sampling=args.sampling,
            skip_steps=args.skip_steps,
            rep_penalty=args.rep_penalty,
            end_penalty=args.end_penalty,
            pktend_bonus=args.pktend_bonus
        )
        for i,p in enumerate(flow,1):
            print(f"Packet {i}: {p}")

if __name__=='__main__':
    main()


"""
(pytorch_env) jmasch@Johnathons-MacBook-Air-3812 seniorDesignPKT-1-current % python transformer/test_model6.py \
  --checkpoint ./best_model7_db_2.pth \
  --prompt "simulate a TCP flow src_ip:10.0.0.5 dst_ip:172.16.1.10 src_port:57506 dst_port:5000 protocol:TCP" \
  --mode flow --min-packets 10 --max-tries 10 \
  --sampling topp --topp 0.92 --temp 1.3 \
  --skip-steps 12 --rep-penalty 0.85 \
  --end-penalty 0.4 --pktend-bonus 1.3
"""