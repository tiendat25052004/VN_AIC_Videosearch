FROM python:3.11

# Create app directory
WORKDIR /app/AIC2024
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Install app dependencies
COPY requirements.txt /app/AIC2024/

RUN pip install git+https://github.com/openai/CLIP.git
RUN pip install -r requirements.txt

# Bundle app source
COPY . /app/AIC2024

EXPOSE 8080
CMD [ "python3", "app.py" ]