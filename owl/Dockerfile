FROM --platform=linux/amd64 pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime AS trace_owl
# Build context: the root of the FaithMIL repository (see do_build.sh), so that trace_mil/ can be copied.

ENV PYTHONUNBUFFERED=1
# No internet on Grand Challenge: never try to download weights at runtime
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    MPLCONFIGDIR=/tmp/matplotlib

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

RUN python -m venv --system-site-packages --without-pip /home/user/venv
ENV PATH="/home/user/venv/bin:$PATH"

COPY --chown=user:user owl/requirements.txt /opt/app/

RUN python -m pip install \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user trace_mil /opt/app/trace_mil
COPY --chown=user:user owl/app.py /opt/app/
COPY --chown=user:user owl/inference.py owl/owl_model.py owl/owl_report.py /opt/app/

LABEL org.grand-challenge.api-method="invoke"

ENTRYPOINT ["python", "app.py"]
