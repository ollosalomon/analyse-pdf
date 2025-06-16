# Entrée CLI ou script principal
import sys

if __name__ == "__main__":
    mode = input("Choisissez le mode (cli / web) : ").strip().lower()
    if mode == "cli":
        from app.interface.cli import main as cli_main
        cli_main()
    elif mode == "web":
        import subprocess
        subprocess.run(["streamlit", "run", "app/interface/streamlit_app.py"])
    else:
        print("Mode non reconnu. Utilisez 'cli' ou 'web'.")
        sys.exit(1)
