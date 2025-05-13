#!/usr/bin/env python3
"""
Test script to verify that imports work correctly.
"""
import os
import sys
from pathlib import Path

print("Current directory:", os.getcwd())
print("Python path:", sys.path)

try:
    print("\nTrying direct model import...")
    from models.llm_interface import get_llm
    print("✅ Direct model import successful!")
except ImportError as e:
    print(f"❌ Direct model import failed: {e}")
    
try:
    print("\nTrying absolute model import...")
    from autonomous_agent.models.llm_interface import get_llm
    print("✅ Absolute model import successful!")
except ImportError as e:
    print(f"❌ Absolute model import failed: {e}")
    
try:
    print("\nTrying logger import...")
    from utils.logger import setup_logger
    print("✅ Logger import successful!")
except ImportError as e:
    print(f"❌ Logger import failed: {e}")
    
try:
    print("\nTrying config import...")
    from config import MODEL_CONFIG
    print("✅ Config import successful!")
except ImportError as e:
    print(f"❌ Config import failed: {e}")
    
print("\nTest complete.") 