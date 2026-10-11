#!/bin/bash
#
# Runs inside the shared environment image (environment/Dockerfile). pytest is baked into
# that image, so do not install or download packages at verify time.
#
# Keep ground truth under tests/ (not environment/, where the agent could read them).
# --ctrf writes a standard JSON report; write 1/0 to /logs/verifier/reward.txt.
#
# Always write reward.txt. Snapshot grader hashes, make /tests root-only, re-run
# analyze.py as user nobody (so it cannot read the grader or write reward.txt),
# restore /tests mode, refuse to grade if hashes changed, then pytest. Overwrite
# reward.txt only from this script after pytest.

mkdir -p /logs/verifier /tmp/agent_run /root/vf_hidden
chmod 700 /logs/verifier /root /root/vf_hidden
rm -f /logs/verifier/reward.txt

python3 /tests/restore_heldout.py
restore_status=$?

integrity_status=1
analyze_status=1
locked=0

python3 - <<'PY'
import hashlib
import json
import os

root = "/tests"
digests = {}
for dirpath, _, files in os.walk(root):
    for name in sorted(files):
        path = os.path.join(dirpath, name)
        if os.path.islink(path) or not os.path.isfile(path):
            continue
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        digests[os.path.relpath(path, root)] = h.hexdigest()
os.makedirs("/root/vf_hidden", exist_ok=True)
with open("/root/vf_hidden/grader_sha256.json", "w") as f:
    json.dump(digests, f, sort_keys=True)
PY
hash_status=$?

if [ "$restore_status" -eq 0 ] && [ "$hash_status" -eq 0 ] && [ -f /app/output/analyze.py ] && [ ! -L /app/output/analyze.py ]; then
  rm -rf /tmp/agent_run
  mkdir -p /tmp/agent_run
  cp /app/output/analyze.py /tmp/agent_run/analyze.py
  # Sibling helper modules travel with analyze.py, as instruction.md promises.
  for f in /app/output/*.py; do
    [ -f "$f" ] || continue
    [ -L "$f" ] && continue
    base=$(basename "$f")
    [ "$base" = "analyze.py" ] && continue
    case "$base" in
      sitecustomize.py|usercustomize.py) continue ;;
    esac
    cp "$f" "/tmp/agent_run/$base"
  done
  chmod -R a+rX /tmp/agent_run
  # solve.sh (oracle) and the agent both write JSON as root; nobody cannot
  # overwrite a 644 root-owned file even in a 777 directory.
  chmod a+rwx /app/output
  chmod -R a+rw /app/output
  chmod -R a+rX /app/data
  # Drop any JSON written on the calibration extract. pytest must grade files
  # this re-run wrote from the restored /app/data, not stale pre-seeded outputs.
  rm -f /app/output/adjustment_set.json /app/output/effect_estimate.json /app/output/customer_predictions.json
  touch /tmp/agent_run/.grade_start

  chmod -R 700 /tests
  if [ "$(stat -c %a /tests)" = "700" ]; then
    locked=1
    python3 - <<'PY'
import os
import pwd
import subprocess
import sys

pw = pwd.getpwnam("nobody")


def drop_privs():
    os.environ["HOME"] = "/tmp/agent_run"
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ.pop("PYTHONPATH", None)
    os.setgid(pw.pw_gid)
    os.setuid(pw.pw_uid)


# -E -s: ignore PYTHON* env vars and the user site-packages, like -I, but keep
# the script's own directory on sys.path. -I would also imply -P, which drops
# it, so a helper module placed beside analyze.py exactly as instruction.md
# directs ("keep any helper module as a plain .py file beside it") would raise
# ModuleNotFoundError under grading even though it imports fine in /app/output.
# The sibling .py files were copied into /tmp/agent_run above; running from
# there (not /app/output) with sitecustomize/usercustomize excluded keeps the
# isolation the copy was meant to provide.
try:
    proc = subprocess.run(
        [sys.executable, "-E", "-s", "analyze.py"],
        cwd="/tmp/agent_run",
        timeout=180,
        preexec_fn=drop_privs,
    )
    sys.exit(proc.returncode)
except subprocess.TimeoutExpired:
    sys.exit(1)
PY
    analyze_status=$?
    if [ "$analyze_status" -eq 0 ]; then
      python3 - <<'PY'
import os
import sys

sentinel = os.path.getmtime("/tmp/agent_run/.grade_start")
for name in ("adjustment_set.json", "effect_estimate.json", "customer_predictions.json"):
    path = f"/app/output/{name}"
    if os.path.islink(path) or not os.path.isfile(path):
        sys.exit(1)
    if os.path.getmtime(path) < sentinel:
        sys.exit(1)
PY
      analyze_status=$?
    fi
  else
    analyze_status=1
  fi
fi

if [ "$locked" -eq 1 ] || [ -d /tests ]; then
  chmod -R u+rwX,go+rX /tests 2>/dev/null || true
fi

if [ -d /tests ] && [ -f /root/vf_hidden/grader_sha256.json ]; then
  python3 - <<'PY'
import hashlib
import json
import os
import sys

with open("/root/vf_hidden/grader_sha256.json") as f:
    expected = json.load(f)
root = "/tests"
got = {}
for dirpath, _, files in os.walk(root):
    for name in sorted(files):
        path = os.path.join(dirpath, name)
        if os.path.islink(path) or not os.path.isfile(path):
            continue
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        got[os.path.relpath(path, root)] = h.hexdigest()
if got != expected:
    sys.exit(1)
PY
  integrity_status=$?
else
  integrity_status=1
fi

rm -f /logs/verifier/reward.txt

pytest_status=1
if [ "$integrity_status" -eq 0 ]; then
  pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA
  pytest_status=$?
fi

if [ "$restore_status" -eq 0 ] && [ "$analyze_status" -eq 0 ] && [ "$integrity_status" -eq 0 ] && [ "$pytest_status" -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
