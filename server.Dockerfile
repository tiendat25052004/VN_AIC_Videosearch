FROM python:3.11

# Create app directory
WORKDIR /app/AIC2024
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
ENV TORCH_FORCE_WEIGHTS_ONLY_LOAD=true
# Install app dependencies
COPY requirements.txt /app/AIC2024/

RUN pip install git+https://github.com/openai/CLIP.git
RUN pip install -r requirements.txt
RUN pip install gevent
# Bundle app source
COPY . /app/AIC2024

EXPOSE 8080
CMD [ "python3", "app.py" ]