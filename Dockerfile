FROM anibali/pytorch:1.5.0-cuda10.2
# FROM docker.bic.mni.mcgill.ca/pytorch-cuda:1.7-cu101

RUN conda update conda

COPY app/requirements.txt requirements.txt

RUN conda install -c anaconda pip && pip install -r requirements.txt

COPY app/ /app/

USER user

RUN sudo addgroup --gid 618 noel && sudo usermod -a -G noel user
# RUN sudo usermod -a -G noel user

RUN sudo chown -R user:noel .  && sudo chmod -R 775 . && sudo chmod -R g+s .

CMD ["python3", "inference.py"]
