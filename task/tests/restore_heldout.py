"""Copy verifier-owned held-out tables onto /app/data before re-running analyze.py."""
import hashlib
import os
import shutil

FIXTURE_DIR = "/tests/fixtures/heldout"
DATA_DIR = "/app/data"

EXPECTED_SHA256 = {
    "assignment_revisions.csv": "5fad7fc8cab6d65cf6795be67a1ea42f9d585136835668389106ca365d79b373",
    "customers.csv": "3ebc626dccbe1ec3af32dfcfa22ea703649b9d37368b16dbfab1385c86a7cf66",
    "data_dictionary.md": "0c1b390a08419c877aeedf100683322d8d85012e8ec1aeb57880c85cc2166478",
    "plan_events.csv": "ac8279dbf034fde38c2e8d31e1fa9c18f0958a1ea8d06bb6a112e225ef91de41",
    "support_tickets.csv": "b352ae83eb812e385f1be79792c2e62fd10ed14b4b8107967491db8547bb26e9",
    "usage_summary.csv": "c00e7e9e4813eeefa85e2eccbd2d0451ba7b579b8924d979ffc1f774a9aae5a2",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, digest in EXPECTED_SHA256.items():
        src = os.path.join(FIXTURE_DIR, name)
        if os.path.islink(src) or not os.path.isfile(src):
            raise SystemExit(f"missing trusted fixture: {src}")
        got = sha256_file(src)
        if got != digest:
            raise SystemExit(f"fixture hash mismatch for {name}: {got}")
        dst = os.path.join(DATA_DIR, name)
        shutil.copy2(src, dst)


if __name__ == "__main__":
    main()
