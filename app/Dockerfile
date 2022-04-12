FROM python:3

# library
RUN pip install --upgrade pip && pip install streamlit

# change dir
WORKDIR /usr/src/app

# run streamlit
CMD ["streamlit", "run", "main.py"]
