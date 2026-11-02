"""Fetch only a kernel's log (one API call; the CLI pages through every output file -> 429).

    <python with the kaggle package> scripts/fetch_log.py <owner>/<slug> <out.log>
Prints the [hybrid ...] lines and errors.
"""
import json
import sys
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

ref, out = sys.argv[1], sys.argv[2]
api = KaggleApi()
api.authenticate()
with api.build_kaggle_client() as k:
    r = ApiListKernelSessionOutputRequest()
    r.user_name, r.kernel_slug = ref.split("/")
    r.page_size = 1
    resp = k.kernels.kernels_api_client.list_kernel_session_output(r)
Path(out).write_text(resp.log or "")
try:
    text = "".join(e.get("data", "") for e in json.loads(resp.log or "[]"))
except ValueError:
    text = resp.log or ""
seen = set()
for line in text.splitlines():
    if line in seen:
        continue
    seen.add(line)
    if any(k in line for k in ("[hybrid", "Traceback", "Error", "SMOKE", "server ready", "vllm ")):
        print(line[:400])
