# Gaussia Lambda Container
# Builds from local wheel (gaussia is not available on PyPI)

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

# Install numpy and scipy from pre-built manylinux wheels first
# These versions have pre-built wheels that don't require compilation
RUN pip install "numpy==1.26.4" "scipy==1.14.1" --target "${LAMBDA_TASK_ROOT}"

# Create constraints file to prevent scipy/numpy upgrades
RUN echo "numpy==1.26.4" > /tmp/constraints.txt && \
    echo "scipy==1.14.1" >> /tmp/constraints.txt

# Install the wheel with the specified extras (MODULE_EXTRA build arg)
ARG MODULE_EXTRA=generators
RUN WHEEL=$(ls /tmp/*.whl) && pip install "${WHEEL}[${MODULE_EXTRA}]" --target "${LAMBDA_TASK_ROOT}" -c /tmp/constraints.txt

# Install additional runtime dependencies
COPY examples/generators/aws-lambda/requirements.txt .
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy Lambda handler and business logic
COPY examples/generators/aws-lambda/handler.py examples/generators/aws-lambda/run.py ${LAMBDA_TASK_ROOT}/

# Set the Lambda handler
CMD ["handler.lambda_handler"]
