import sys
import isolated_config
from streamlit.web import cli

isolated_config.require_target("mysql")
isolated_config.require_target("redis")
sys.argv = ["streamlit", "run", "/app/main.py", "--server.address=0.0.0.0",
            "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
sys.exit(cli.main())
