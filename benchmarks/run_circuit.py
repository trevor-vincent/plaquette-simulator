import time
from plaquette.circuit import Circuit
from plaquette.device import Device

import sys

if len(sys.argv) != 3:
    print("Usage: python3 run_circuit.py <circuit_file> <num_samples>")
    exit(1)

with open(sys.argv[1], 'r') as file:
    content = file.read()

circuit = Circuit.from_str(content)
num_samples = int(sys.argv[2])
dev = Device("clifford")
dev.run(circuit)

start = time.time()
for _ in range(num_samples):
    dev.get_sample()
end = time.time()

print(sys.argv[1], end - start)
