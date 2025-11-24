@echo off
echo === Setting up Python 3.10 Virtual Environment ===

py -3.10 -m venv venv
call venv\Scripts\activate

echo === Installing Requirements ===
pip install -r requirements.txt

echo === Installing Llama-Cpp-Python ===
pip install llama-cpp-python --prefer-binary

echo === Downloading Llama 3.1 GGUF Model ===
echo Please set HUGGINGFACE_TOKEN environment variable before running this script.
powershell -Command "Invoke-WebRequest https://huggingface.co/unsloth/llama-3.1-8b-instruct-gguf/resolve/main/Llama-3.1-8B-Instruct-Q4_K_M.gguf -Headers @{Authorization='Bearer $env:HUGGINGFACE_TOKEN'} -OutFile models\llama3.gguf"

echo === Setup Complete ===
pause
