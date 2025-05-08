#!/usr/bin/env python3
import dpkt
import socket
import json
import os
import sys
import sqlite3
from datetime import datetime
from collections import defaultdict

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

def canonical_flow_key(src_ip, dst_ip, src_port, dst_port, protocol):
    """Returns a tuple that is the same for both directions of a flow.
    This allows the same key to be used regardless of the direction of the flow (in theory)."""
    # the src_ip, src_port tuple will always be less than the dst_ip, dst_port tuple
    if (src_ip, src_port) < (dst_ip, dst_port):
        return (src_ip, dst_ip, src_port, dst_port, protocol)
    else:
        return (dst_ip, src_ip, dst_port, src_port, protocol)

def init_db(db_path):
    """Initialize the SQLite database and create the streams table if it does not exist."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS streams (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT,
            prompt TEXT,
            tags TEXT,
            src_ip TEXT,
            dst_ip TEXT,
            src_port INTEGER,
            dst_port INTEGER,
            protocol TEXT,
            packets TEXT
        )
    ''')
    conn.commit()
    return conn

# might have to limit the number of packets
def process_pcap(pcap_file_path, description, prompt, tags, db_path="streams.db", max_packets=100_000_000):
    packet_streams = defaultdict(list)

    with open(pcap_file_path, 'rb') as f:
        pcap_reader = dpkt.pcap.Reader(f)

        for i, (ts, buf) in enumerate(pcap_reader):
            eth = dpkt.ethernet.Ethernet(buf)
            if isinstance(eth.data, dpkt.ip.IP):
                ip = eth.data
                src_ip = socket.inet_ntoa(ip.src)
                dst_ip = socket.inet_ntoa(ip.dst)

                if isinstance(ip.data, (dpkt.tcp.TCP, dpkt.udp.UDP)):
                    transport_layer = ip.data
                    protocol = 'TCP' if isinstance(transport_layer, dpkt.tcp.TCP) else 'UDP'
                    src_port = transport_layer.sport
                    dst_port = transport_layer.dport
                    payload_size = len(transport_layer.data)
                    payload_hex = [f"{byte:02x}" for byte in transport_layer.data]
                    timestamp = datetime.fromtimestamp(ts).isoformat()

                    # TCP flags
                    flags = ""
                    if protocol == 'TCP':
                        tcp_flags = {
                            dpkt.tcp.TH_FIN: "FIN",
                            dpkt.tcp.TH_SYN: "SYN",
                            dpkt.tcp.TH_RST: "RST",
                            dpkt.tcp.TH_PUSH: "PUSH",
                            dpkt.tcp.TH_ACK: "ACK",
                            dpkt.tcp.TH_URG: "URG",
                        }
                        flags = ",".join(name for bit, name in tcp_flags.items() if transport_layer.flags & bit)

                    # Create a unique identifier for each stream
                    stream_id = canonical_flow_key(src_ip, dst_ip, src_port, dst_port, protocol)
                    packet = {
                        "src_ip": src_ip,
                        "dst_ip": dst_ip,
                        "protocol": protocol,
                        "src_port": src_port,
                        "dst_port": dst_port,
                        "timestamp": timestamp,
                        "flags": flags,
                        "payload_size": payload_size,
                        "payload_hex": payload_hex,
                    }
                    packet_streams[stream_id].append(packet)

            if i >= max_packets - 1:
                break

    output_streams = []
    for stream_id, packets in packet_streams.items():
        # input minimum packet stream length here
        if len(packets) >= 10:
            output_streams.append({
                "description": description,
                "prompt": prompt,
                "tags": tags,
                "stream_id": {
                    "src_ip": stream_id[0],
                    "dst_ip": stream_id[1],
                    "src_port": stream_id[2],
                    "dst_port": stream_id[3],
                    "protocol": stream_id[4],
                },
                "packets": packets
            })

    conn = init_db(db_path)
    c = conn.cursor()
    for stream in output_streams:
        sid = stream["stream_id"]
        # Store the packets as a JSON string
        packets_json = json.dumps(stream["packets"])
        c.execute('''
            INSERT INTO streams (description, prompt, tags, src_ip, dst_ip, src_port, dst_port, protocol, packets)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            stream["description"],
            stream["prompt"],
            stream["tags"],
            sid["src_ip"],
            sid["dst_ip"],
            sid["src_port"],
            sid["dst_port"],
            sid["protocol"],
            packets_json
        ))
    conn.commit()
    conn.close()
    print(f"Appended {len(output_streams)} streams from {pcap_file_path} to the database at {db_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, '..')
    db_path = os.path.join(script_dir, "streams.db")

    # List of PCAP files to process
    # TODO: implement use of 'description' and 'tags' fields in model
    pcap_files = [
        # {"path": os.path.join(data_dir, "maccdc2012_00000.pcap"), "description": "TCP and UDP packet data", "tags": "Normal,DNS"},
        {"path": os.path.join(data_dir, "pcaps2.pcap"), "description": "TCP packet data", "tags": "DNS"},
    ]

    for pcap_info in pcap_files:
        process_pcap(
            pcap_file_path=pcap_info["path"],
            description=pcap_info["description"],
            prompt=f"Generate traffic for {pcap_info['description']}",
            tags=pcap_info["tags"],
            db_path=db_path
        )
