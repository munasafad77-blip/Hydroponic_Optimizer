#!/bin/bash
set -e
cd "$(dirname "$0")"
clear
echo "============================================================"
echo "  Hydroponic Medium Optimizer"
echo "============================================================"
echo

if ! command -v python3 &> /dev/null; then
    echo "This computer needs Python 3 installed first."
    echo "Go to https://www.python.org/downloads/ , install it,"
    echo "then run this again."
    read -p "Press Enter to close..."
    exit 1
fi

NEED_SETUP=0
if [ ! -f ".venv/bin/python" ]; then
    NEED_SETUP=1
elif ! ".venv/bin/python" -c "import streamlit" &> /dev/null; then
    NEED_SETUP=1
fi

if [ "$NEED_SETUP" = "1" ]; then
    echo "Setting up for the first time -- please wait,"
    echo "this can take a few minutes. You will only see this once."
    echo
    rm -f setup_log.txt
    python3 -m venv .venv >> setup_log.txt 2>&1
    ".venv/bin/python" -m pip install --upgrade pip >> setup_log.txt 2>&1
    if ! ".venv/bin/python" -m pip install -r requirements.txt >> setup_log.txt 2>&1; then
        echo
        echo "Something went wrong during setup."
        echo "Please send the file setup_log.txt from this folder"
        echo "so it can be looked at."
        read -p "Press Enter to close..."
        exit 1
    fi
    echo "Setup complete."
    echo
fi

echo "Starting the app -- your browser will open automatically"
echo "in a few seconds."
echo
echo "IMPORTANT: keep this window open while you use the app."
echo "When you are done, close this window or press Ctrl+C."
echo "============================================================"
echo

".venv/bin/python" -m streamlit run app.py > app_log.txt 2>&1

echo
echo "The app has stopped."
read -p "Press Enter to close..."
