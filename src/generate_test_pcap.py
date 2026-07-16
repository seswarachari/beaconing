import os
from scapy.all import Ether, IP, TCP, wrpcap
import time
import random

def generate_beacon_pcap(output_file, num_beacons=20, interval=30):
    print(f"Generating test PCAP with {num_beacons} beacons every ~{interval}s...")
    packets = []
    
    # Base timestamp
    base_time = time.time() - (num_beacons * interval)
    
    # 1. Generate some benign web traffic
    print("Generating benign web traffic...")
    for _ in range(50):
        pkt_time = base_time + random.uniform(0, num_beacons * interval)
        src_port = random.randint(1024, 65535)
        # Web request
        req = Ether()/IP(src="192.168.1.50", dst="93.184.216.34")/TCP(sport=src_port, dport=443, flags="PA")/"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"
        req.time = pkt_time
        packets.append(req)
        
        # Web response
        resp = Ether()/IP(src="93.184.216.34", dst="192.168.1.50")/TCP(sport=443, dport=src_port, flags="PA")/("A" * random.randint(500, 1500))
        resp.time = pkt_time + random.uniform(0.01, 0.5)
        packets.append(resp)
        
    # 2. Generate a fixed beacon (C2 traffic)
    print("Generating fixed C2 beaconing traffic...")
    c2_ip = "185.123.45.67"
    for i in range(num_beacons):
        # Exactly 'interval' seconds apart
        pkt_time = base_time + (i * interval)
        src_port = random.randint(1024, 65535)
        
        # Beacon request
        req = Ether()/IP(src="192.168.1.100", dst=c2_ip)/TCP(sport=src_port, dport=443, flags="PA")/"BEACON_DATA"
        req.time = pkt_time
        packets.append(req)
        
        # Beacon response
        resp = Ether()/IP(src=c2_ip, dst="192.168.1.100")/TCP(sport=443, dport=src_port, flags="PA")/"OK"
        resp.time = pkt_time + 0.05
        packets.append(resp)
        
    # Sort packets by time
    packets.sort(key=lambda p: p.time)
    
    # Write to PCAP
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    wrpcap(output_file, packets)
    print(f"Saved {len(packets)} packets to {output_file}")

if __name__ == '__main__':
    out_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_pcap', 'test_beacon.pcap')
    generate_beacon_pcap(out_path, num_beacons=30, interval=60)
