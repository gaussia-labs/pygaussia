# Gaussia Lambda Container
# Builds from local wheel (gaussia is not available on PyPI)
#
# IMPORTANT: This Dockerfile is designed to be built with the repo root as context.
# Build command: docker build -t gaussia-role-adherence -f examples/role_adherence/aws-lambda/Dockerfile --build-arg MODULE_EXTRA=role_adherence .

FROM public.ecr.aws/lambda/python:3.11 AS builder

# Install build tools for Python wheel
RUN pip install --upgrade pip build

# Copy the gaussia source and build wheel
WORKDIR /build
COPY gaussia/ ./gaussia/
COPY pyproject.toml README.md LICENSE ./
RUN python -m build --wheel --outdir /dist

# Final Lambda image
FROM public.ecr.aws/lambda/python:3.11

# Copy built wheel from builder stage
COPY --from=builder /dist/*.whl /tmp/

# Upgrade pip
RUN pip install --upgrade pip

# Pre-install numpy and scipy with manylinux wheels (avoids compilation issues)
# The Lambda base image lacks compilers needed to build these from source.
# These versions have pre-built wheels compatible with the Lambda environment.
RUN pip install "numpy==1.26.4" "scipy==1.14.1" --target "${LAMBDA_TASK_ROOT}"

# Create constraints file to prevent scipy/numpy from being upgraded
# (gaussia requires scipy>=1.14.1, but 1.14.1 is the last version with manylinux wheels
# that don't require compilation on Lambda's older glibc)
RUN echo "numpy==1.26.4" > /tmp/constraints.txt && \
    echo "scipy==1.14.1" >> /tmp/constraints.txt

# Install the wheel with the specified extras (MODULE_EXTRA build arg)
# Note: WHEEL=$(ls /tmp/*.whl) captures full path, so don't prefix with /tmp/
ARG MODULE_EXTRA=role_adherence
RUN WHEEL=$(ls /tmp/*.whl) && pip install "${WHEEL}[${MODULE_EXTRA}]" --target "${LAMBDA_TASK_ROOT}" -c /tmp/constraints.txt

# Install additional runtime dependencies (LLM providers)
COPY examples/role_adherence/aws-lambda/requirements.txt .
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy Lambda handler and business logic
COPY examples/role_adherence/aws-lambda/handler.py examples/role_adherence/aws-lambda/run.py ${LAMBDA_TASK_ROOT}/

# Set the Lambda handler
CMD ["handler.lambda_handler"]
