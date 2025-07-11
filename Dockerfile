FROM amazonlinux:2023 AS builder
RUN yum groupinstall -y "Development Tools" && \
    yum install -y gcc libffi-devel openssl-devel python3 python3-pip make zip

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    pip install -r requirements.txt -t /package

FROM public.ecr.aws/sam/build-python3.13 AS final
COPY --from=builder /package /var/task
COPY bot_trading.py /var/task
