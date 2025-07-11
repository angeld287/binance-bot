FROM amazonlinux:2023 AS builder
RUN yum groupinstall -y "Development Tools" && \
    yum install -y gcc libffi-devel openssl-devel python3 python3-pip make zip python3-devel

COPY requirements.txt .

RUN pip install --no-binary=cffi -r requirements.txt -t /package
    
RUN python3 -c "import sys; import _cffi_backend; print('_cffi_backend found at:', _cffi_backend.__file__)"

RUN find / -type f -name "_cffi_backend*.so" 2>/dev/null || { echo "_cffi_backend.so NOT FOUND"; exit 1; }


FROM public.ecr.aws/sam/build-python3.13 AS final
WORKDIR /var/task
COPY --from=builder /package /var/task
COPY bot_trading.py /var/task
