"""Moves every weight a little and takes a moment over it, like real training."""

import json
import os
import struct
import time
from array import array

with open(os.environ["FEDTALLY_BASE"], "rb") as f:
    size = struct.unpack("<Q", f.read(8))[0]
    header = f.read(size)
    weights = array("f", f.read())

for name in json.loads(header):
    if name == "__metadata__":
        continue
    weights = array("f", (w + 0.01 for w in weights))

with open(os.environ["FEDTALLY_OUT"], "wb") as f:
    f.write(struct.pack("<Q", size))
    f.write(header)
    weights.tofile(f)

time.sleep(2)
print(1)
