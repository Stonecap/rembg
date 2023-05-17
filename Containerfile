FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

RUN mkdir -p ~/.u2net
RUN wget https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_cloth_seg.onnx -O ~/.u2net/u2net_cloth_seg.onnx

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./main.py /app
COPY ./rembg /app/rembg
