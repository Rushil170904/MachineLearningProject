from scapy.all import rdpcap, TCP, IP
import pandas as pd
import math
from collections import defaultdict, Counter
import os

folder = "NonVPN-PCAPs-01"
output_csv = "nonVpn_features.csv"


def entropy(values):
    counts = Counter(values)
    total = len(values)
    return -sum((count/total) * math.log2(count/total) for count in counts.values())

def extract_features_scapy(pcap_file):
    packets = rdpcap(pcap_file)   # loads packets into memory
    flows = defaultdict(lambda: {
        "fwd_packets": 0, "bwd_packets": 0,
        "fwd_bytes": 0, "bwd_bytes": 0,
        "pkt_sizes": [], "timestamps": [],
        "syn": 0, "fin": 0, "rst": 0, "ack": 0
    })

    for pkt in packets:
        if not pkt.haslayer(IP):
            continue

        ip = pkt[IP]
        proto = "TCP" if pkt.haslayer(TCP) else "OTHER"

        src, dst = ip.src, ip.dst
        sport = pkt[TCP].sport if pkt.haslayer(TCP) else 0
        dport = pkt[TCP].dport if pkt.haslayer(TCP) else 0

        key = (src, dst, sport, dport, proto)
        length = len(pkt)
        ts = pkt.time

        flows[key]["pkt_sizes"].append(length)
        flows[key]["timestamps"].append(ts)

        if src < dst:
            flows[key]["fwd_packets"] += 1
            flows[key]["fwd_bytes"] += length
        else:
            flows[key]["bwd_packets"] += 1
            flows[key]["bwd_bytes"] += length

        if pkt.haslayer(TCP):
            tcp = pkt[TCP]
            if tcp.flags & 0x02:  # SYN
                flows[key]["syn"] += 1
            if tcp.flags & 0x01:  # FIN
                flows[key]["fin"] += 1
            if tcp.flags & 0x04:  # RST
                flows[key]["rst"] += 1
            if tcp.flags & 0x10:  # ACK
                flows[key]["ack"] += 1

    rows = []
    for k, v in flows.items():
        duration = max(v["timestamps"]) - min(v["timestamps"]) if len(v["timestamps"]) > 1 else 0.0
        rows.append({
            "src_ip": k[0], "dst_ip": k[1],
            "src_port": k[2], "dst_port": k[3],
            "protocol": k[4],
            "fwd_packets": v["fwd_packets"],
            "bwd_packets": v["bwd_packets"],
            "fwd_bytes": v["fwd_bytes"],
            "bwd_bytes": v["bwd_bytes"],
            "avg_pkt_size": sum(v["pkt_sizes"]) / len(v["pkt_sizes"]) if v["pkt_sizes"] else 0,
            "pkt_size_entropy": entropy(v["pkt_sizes"]) if v["pkt_sizes"] else 0,
            "duration": duration,
            "pkt_rate": (len(v["pkt_sizes"]) / duration) if duration > 0 else 0,
            "byte_rate": ((v["fwd_bytes"] + v["bwd_bytes"]) / duration) if duration > 0 else 0,
            "syn_count": v["syn"], "fin_count": v["fin"],
            "rst_count": v["rst"], "ack_count": v["ack"]
        })
    return pd.DataFrame(rows)

first_file = True

for file in os.listdir(folder):
    if file.endswith(".pcap"):
        filepath = os.path.join(folder, file)
        print(f"Processing {filepath} ...")

        df = extract_features_scapy(filepath)

        # Append to CSV
        df.to_csv(output_csv, mode='a', header=first_file, index=False)

        # Only first file writes header
        first_file = False

print("✅ Feature extraction complete. All results saved to:", output_csv)