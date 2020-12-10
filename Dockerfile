FROM docker.bic.mni.mcgill.ca/pytorch-cuda:1.7-cu101

RUN conda update conda

WORKDIR /app
RUN chmod -R a+w .

COPY app/* /app
CMD ["python3", "main.py"]
