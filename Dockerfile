FROM ubuntu:22.04


WORKDIR /MalSynGen


RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    unzip \
    && rm -rf /var/lib/apt/lists/*


RUN rm /bin/sh && ln -s /bin/bash /bin/sh


RUN pip3 install pipenv
RUN pip install urllib3

COPY ./ /MalSynGen/

# Install dependencies using pipenv and requirements.txt

RUN pipenv install --skip-lock --dev && \
    pipenv install -r requirements.txt
RUN pip3 install requests
RUN pip install selenium
RUN chmod +x /MalSynGen/scripts/run_app_in_docker.sh
RUN /MalSynGen/scripts/run_app_in_docker.sh



