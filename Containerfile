FROM python:3.10

WORKDIR /code

COPY . .

RUN pip install .

RUN mkdir -p ~/.u2net
RUN wget https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_cloth_seg.onnx -O ~/.u2net/u2net_cloth_seg.onnx

EXPOSE 5000
ENTRYPOINT [ "rembg" ]E
CMD ["s"]
