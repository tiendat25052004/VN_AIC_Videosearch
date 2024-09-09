FROM python:3.12

# Create app directory
WORKDIR /app

# Install app dependencies
COPY requirements.txt ./

RUN pip install -r requirements.txt
RUN pip install flask
RUN pip install flask-cors
RUN pip install flask-socketio
RUN pip install pyngrok==4.1.1
# Bundle app source
COPY . .

EXPOSE 5000
CMD [ "python3", "appStorage.py" ]