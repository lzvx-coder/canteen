import os

from backend.app import run


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8004))
    run(host="0.0.0.0", port=port, open_browser=False, auto_shutdown=False)
