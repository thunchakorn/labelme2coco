FROM python:3.8-slim-buster
WORKDIR /app

COPY labelme2coco.py /
COPY ./ ./

RUN python -m pip install -r requirements.txt
RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev 


# e.g. docker run --rm -it -v ${pwd}/test_polygon:/app/test_polygon labelme2coco
# python labelme2coco.py test_polygon/train --output test_polygon/train.json
