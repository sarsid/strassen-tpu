#!/usr/bin/env python3
"""CLI entry point for the separately tracked experimental Strassen kernel."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from strassen_mm.strassen_optimized import main
if __name__=='__main__':main()
