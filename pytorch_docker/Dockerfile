FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

COPY ./wide_deep_k8s.py /opt/ctr-prediction/
COPY ./requirements.txt /opt/ctr-prediction/

RUN pip3 install -r /opt/ctr-prediction/requirements.txt

ENTRYPOINT ["python3", "/opt/ctr-prediction/wide_deep_k8s.py"]
