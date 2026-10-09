FROM python:3.11-slim

# The benchmark image deliberately contains no questions, answer keys, or
# papers. DockerREPL mounts only the selected frozen corpus read-only at run
# time, so model-authored Python cannot inspect the benchmark labels.
RUN pip install --no-cache-dir \
    pandas==2.2.3 numpy==1.26.4 regex==2024.11.6 tabulate==0.9.0
RUN useradd -m -s /bin/bash sandbox \
    && mkdir -p /workspace/data/corpus \
    && chown -R sandbox:sandbox /workspace

LABEL org.envoybench.sandbox="v1"
ENV CORPUS_DIR=/workspace/data/corpus
USER sandbox
WORKDIR /workspace
CMD ["python3", "-i"]
