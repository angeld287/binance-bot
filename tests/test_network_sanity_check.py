import os
import sys
import subprocess
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from core.network import network_sanity_check


def test_network_sanity_check_and_no_proxy_references():
    # Should not raise even if network is unavailable
    network_sanity_check()

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["rg", "-i", "PROXY_URL", "src"], cwd=repo_root, capture_output=True, text=True
    )
    assert result.stdout.strip() == ""
