FROM vllm/vllm-openai:v0.10.2

# Ensure we do not carry the stock vLLM wheel from the base image.
RUN pip uninstall -y vllm || true

# Copy in the precompiled wheel staged in _dist/ and install it alongside
# any extra Python packages that this fork requires.
COPY cascade/_dist/vllm-0.10.3.dev4+g194ec716c.d20251006.precompiled-cp312-cp312-linux_x86_64.whl /tmp/

RUN pip install --no-cache-dir rapidfuzz \
    && pip install --no-cache-dir /tmp/vllm-0.10.3.dev4+g194ec716c.d20251006.precompiled-cp312-cp312-linux_x86_64.whl \
    && rm -f /tmp/vllm-0.10.3.dev4+g194ec716c.d20251006.precompiled-cp312-cp312-linux_x86_64.whl

ENV VLLM_USE_PRECOMPILED=1
