# Usando a versão exata em que o modelo foi treinado para evitar incompatibilidade
FROM python:3.11-slim

WORKDIR /code

# Copia as dependências (onde você colocará numpy==2.0.2)
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

EXPOSE 80

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
