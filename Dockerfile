FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Use Python 3.11 for better Python perf
# Update the package lists and install necessary dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.11 python3.11-dev

# Set Python 3.11 as the default version (for python3)
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Download get-pip.py script
RUN apt install curl -y
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

# Install pip for Python 3.11
RUN python3 get-pip.py

# Verify Python and pip versions
RUN python3 --version && pip3.11 --version

# Set pip3.11 as the default pip command
RUN update-alternatives --install /usr/bin/pip3 pip3 /usr/local/lib/python3.11/dist-packages/pip 1

ENV PYTHONUNBUFFERED=1

# Install necessary dependencies
# RUN apt-get update && \
#     apt-get install -y python3-pip

# Set the working directory. /app is mounted to the container with -v, 
# but we want to have the right cwd for uvicorn command below
RUN mkdir /app
# WORKDIR /app

# # Copy the app code and requirements filed
# COPY . /app
# COPY requirements.txt .
# WORKDIR $PYSETUP_PATH
COPY ./requirements.txt  /app


COPY ./utils /app/utils
COPY ./static /app/static
COPY ./templates /app/templates
COPY ./infer_server.py /app/infer_server.py
COPY ./download.py /app/download.py

WORKDIR /app


# Install the app dependencies
# RUN pip3 install -r requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
        pip3 install -r requirements.txt

# Expose the FastAPI port
EXPOSE 7860

# Start the FastAPI app using Uvicorn web server
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "14000", "--limit-concurrency", "1000"]
RUN python3 download.py

CMD ["python3", "infer_server.py", "--host=0.0.0.0", "--port=7860", "--model_path=models/sam2ai/whisper-odia-small-finetune-int8-ct2", "--num_workers=2"]


