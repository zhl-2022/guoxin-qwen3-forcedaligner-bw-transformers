#!/usr/bin/env python3
"""Report missing official material; never synthesize AAS or PASS."""
import sys
from submission_entry008 import main
if __name__ == "__main__": raise SystemExit(main(["verify",*sys.argv[1:]]))
