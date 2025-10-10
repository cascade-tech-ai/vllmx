FROM vllm/vllm-openai:v0.11.0

# Ensure we do not carry the stock vLLM wheel from the base image.
RUN pip uninstall -y vllm || true

# Copy in the precompiled wheel staged in _dist/ and install it alongside
# any extra Python packages that this fork requires.
COPY cascade/_dist/vllm-0.11.1.dev9+g7d582f575.precompiled-cp312-cp312-linux_x86_64.whl /tmp/

RUN pip install --no-cache-dir rapidfuzz \
    && pip install --no-cache-dir /tmp/vllm-0.11.1.dev9+g7d582f575.precompiled-cp312-cp312-linux_x86_64.whl \
    && rm -f /tmp/vllm-0.11.1.dev9+g7d582f575.precompiled-cp312-cp312-linux_x86_64.whl

ENV VLLM_USE_PRECOMPILED=1
