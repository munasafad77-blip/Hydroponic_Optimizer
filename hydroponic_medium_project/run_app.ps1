$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (Test-Path .\.venv\Scripts\python.exe) {
    .\.venv\Scripts\python.exe -m streamlit run app.py --server.headless true --server.address 0.0.0.0 --server.port 8501
} else {
    python -m streamlit run app.py --server.headless true --server.address 0.0.0.0 --server.port 8501
}
