FROM anibali/pytorch:1.5.0-cuda10.2

RUN conda update conda

COPY app/ /app/

USER user

RUN sudo addgroup --gid 618 noel && sudo usermod -a -G noel user

RUN sudo chown -R user:noel .  && sudo chmod -R 775 . && sudo chmod -R g+s .

RUN conda install -c anaconda pip && pip install -r requirements.txt

CMD ["python3", "inference.py"]
