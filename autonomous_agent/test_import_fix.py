#!/usr/bin/env python3
"""
Test script to verify that our import fixes work correctly.
This should be run from both the project root directory and from within
the autonomous_agent directory.
"""

import os
import sys
from pathlib import Path

# Add parent directory to sys.path if needed
parent_dir = Path(__file__).parent.parent
if parent_dir not in sys.path:
    sys.path.append(str(parent_dir))

try:
    print("Testing import of llm_interface.get_llm...")
    
    # Try both import styles
    try:
        # Try direct import first (when run from the autonomous_agent directory)
        from models.llm_interface import get_llm
        print("✅ Direct import successful!")
    except ImportError as e:
        print(f"❌ Direct import failed: {e}")
        
        try:
            # Try relative import (when run from a submodule)
            from .models.llm_interface import get_llm
            print("✅ Relative import successful!")
        except ImportError as e:
            print(f"❌ Relative import failed: {e}")
            
            try:
                # Try absolute import (when run from parent directory)
                from autonomous_agent.models.llm_interface import get_llm
                print("✅ Absolute import successful!")
            except ImportError as e:
                print(f"❌ Absolute import failed: {e}")
                
    print("\nAll import tests completed. Check the results above.")
    
except Exception as e:
    print(f"❌ Error during testing: {e}") 