#!/usr/bin/env python3
import sys
sys.path.insert(0, "scripts")
from cactus_wrapper import CactusEmbedder

print("Testing Cactus embedder...")
e = CactusEmbedder("weights/nomic-embed.gguf", "lib/libcactus.so")
v = e.embed("hello world")
print(f"Embedding dim: {len(v)}")
print(f"First 5 values: {v[:5]}")
print("SUCCESS!")
