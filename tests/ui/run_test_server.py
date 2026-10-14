"""Sobe o dashboard Flask em porta de teste com DB e .env TEMPORÁRIOS (nunca toca os reais)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import flask_interface as fi  # noqa: E402

fi.DB_PATH = os.environ["UI_TEST_DB"]
fi.ENV_FILE_PATH = os.environ["UI_TEST_ENV"]

if __name__ == "__main__":
    fi.app.run(host="127.0.0.1", port=int(os.environ.get("UI_TEST_PORT", "5099")), debug=False, threaded=True)
