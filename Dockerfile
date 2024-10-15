FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch-scatter --no-index -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
RUN pip install --no-cache-dir neptune

