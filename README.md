Please do this to initialize the code:

python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt


FOR WINDOWS:

py -3.11 -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

venv\Scripts\Activate.ps1
uvicorn main:app --host 127.0.0.1 --port 5000 --reload

