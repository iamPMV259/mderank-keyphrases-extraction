# Use the official Python 3.10 slim image as the base
FROM python:3.10-slim

# Set environment variables for the virtual environment
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Create the virtual environment
RUN python -m venv $VIRTUAL_ENV

# Upgrade pip in the virtual environment
RUN pip install --upgrade pip

# Install the uv tool (assuming this is the dependency installer you use)
RUN pip install uv --no-cache-dir

# Set the working directory for your application
WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency definitions first to take advantage of Docker caching
COPY pyproject.toml uv.lock ./

# Install dependencies using uv (it will read your pyproject.toml and uv.lock)
RUN uv pip install -r pyproject.toml --all-extras --no-cache-dir

RUN python -m spacy download en_core_web_sm

# Copy the entire project into the container
COPY main.py mderank.py .

# Expose port
EXPOSE 5040

# Set the default command to run your FastAPI app using uvicorn.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5040"]
