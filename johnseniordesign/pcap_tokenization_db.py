
# pcap_tokenization_db.py (fixed version with PacketTokenizer included)
import sqlite3
import random
import re
import torch
from torch.utils.data import Dataset

PACKET_START = "<PACKET_START>"
PACKET_END   = "<PACKET_END>"
PAYLOAD_START= "<PAYLOAD_START>"
PAYLOAD_END  = "<PAYLOAD_END>"
STREAM_START = "<STREAM_START>"
STREAM_END   = "<STREAM_END>"
SOS = "<SOS>"
EOS = "<EOS>"
PAD = "<PAD>"

SPECIAL_TOKENS = [
    PACKET_START, PACKET_END, STREAM_END, STREAM_START,
    PAYLOAD_START, PAYLOAD_END,
    SOS, EOS, PAD
]

def packet_to_tokens(packet_dict):
    tokens = [PACKET_START]
    fields_order = [
        "src_ip", "dst_ip", "protocol", 
        "src_port", "dst_port", "timestamp", 
        "flags", "payload_size"
    ]
    for field in fields_order:
        if field in packet_dict and packet_dict[field] is not None:
            tokens.append(f"{field}:{packet_dict[field]}")
    tokens.append(PACKET_END)
    return tokens

def multi_packets_to_tokens(packet_dicts):
    tokens = [STREAM_START]
    for pkt_dict in packet_dicts:
        tokens.extend(packet_to_tokens(pkt_dict))
    tokens.append(STREAM_END)
    return tokens

def generate_variations(prompt, flow_row):
    src_ip = flow_row["src_ip"]
    dst_ip = flow_row["dst_ip"]
    protocol = flow_row["protocol"]
    flow_id_info = f"src_ip:{src_ip}, dst_ip:{dst_ip}, protocol:{protocol}"
    return [
        f"{prompt} [traffic Details: {flow_id_info}]",
        f"{prompt} with network traffic parameters: {flow_id_info}",
        f"Simulate the following stream: {flow_id_info}. {prompt}",
        f"{prompt}. Details of the packet stream: {flow_id_info}",
        f"Based on the configuration ({flow_id_info}), {prompt}",
        f"Given the network configuration ({flow_id_info}), {prompt}",
        f"Generate a simulation using these network settings: {flow_id_info} → {prompt}",
    ]

class PacketTokenizer:
    def __init__(self):
        self.token2id = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
        self.id2token = {idx: tok for tok, idx in self.token2id.items()}
        self.vocab_size = len(self.token2id)

    def encode(self, token_list):
        encoded = []
        for tok in token_list:
            if tok in self.token2id:
                encoded.append(self.token2id[tok])
            else:
                if "<UNK>" in self.token2id:
                    encoded.append(self.token2id["<UNK>"])
                else:
                    print(f"Unknown token (OOV): {tok}")
        return encoded

    def build_vocab(self, list_of_token_lists):
        unique_tokens = set()
        for token_list in list_of_token_lists:
            unique_tokens.update(token_list)
        for tok in sorted(unique_tokens):
            if tok not in self.token2id:
                self.token2id[tok] = self.vocab_size
                self.id2token[self.vocab_size] = tok
                self.vocab_size += 1

    def decode(self, ids):
        return [self.id2token.get(i, "<UNK>") for i in ids]

class MultiPacketStreamDatasetDB(Dataset):
    def __init__(self, db_path="traffic_data.db", max_packets_per_flow=None, num_prompts_per_flow=4, max_flows=None):
        self.db_path = db_path
        self.max_packets_per_flow = max_packets_per_flow
        self.num_prompts_per_flow = num_prompts_per_flow
        self.max_flows = max_flows
        self.samples = []
        self._load_data()

    def _load_data(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM flows")
        flows = cursor.fetchall()

        if self.max_flows is not None:
            flows = flows[:self.max_flows]

        for flow_row in flows:
            flow_id = flow_row["flow_id"]
            base_prompt = flow_row["prompt"]
            cursor.execute("SELECT * FROM packets WHERE flow_id=? ORDER BY packet_id", (flow_id,))
            packet_rows = cursor.fetchall()
            packet_dicts = [{
                "src_ip":  flow_row["src_ip"],
                "dst_ip":  flow_row["dst_ip"],
                "protocol":flow_row["protocol"],
                "src_port":flow_row["src_port"],
                "dst_port":flow_row["dst_port"],
                "timestamp":row["timestamp"],
                "flags":   row["flags"],
                "payload_size": row["payload_size"]
            } for row in packet_rows]

            if self.max_packets_per_flow is not None:
                packet_dicts = packet_dicts[:self.max_packets_per_flow]

            stream_tokens = multi_packets_to_tokens(packet_dicts)
            prompts = generate_variations(base_prompt, flow_row)
            selected_prompts = random.sample(prompts, min(self.num_prompts_per_flow, len(prompts)))

            for prompt_variant in selected_prompts:
                self.samples.append((prompt_variant, stream_tokens))

        conn.close()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
