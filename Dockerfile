FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime

RUN conda update conda

WORKDIR /workspace
RUN chmod -R a+w .

COPY app/* /workspace/app
CMD ["python", "./app/converter.py"]
