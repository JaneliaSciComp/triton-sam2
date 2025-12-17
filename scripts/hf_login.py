#!/usr/bin/env python3
"""
HuggingFace Login Helper

Prompts for HuggingFace token and stores credentials.
"""

from huggingface_hub import login

print("=" * 70)
print("HuggingFace Authentication")
print("=" * 70)
print()
print("To download SAM3, you need a HuggingFace token.")
print()
print("Steps:")
print("  1. Get your token from: https://huggingface.co/settings/tokens")
print("  2. Request access to SAM3: https://huggingface.co/facebook/sam3")
print("  3. Paste your token below")
print()
print("-" * 70)
print()

# Login (will prompt for token)
login()

print()
print("✓ Login successful!")
print()
print("Next steps:")
print("  1. Request access to SAM3 at: https://huggingface.co/facebook/sam3")
print("  2. Run: pixi run setup-sam3")
