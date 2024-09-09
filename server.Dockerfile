FROM python:3.12

# Create app directory
WORKDIR /app/AIC2024

# Install app dependencies
COPY requirements.txt ./

RUN pip install -r requirements.txt

# Bundle app source
COPY . .

EXPOSE 8080
CMD [ "python", "app.py" ]