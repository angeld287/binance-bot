FROM public.ecr.aws/sam/build-python3.13 AS builder
COPY requirements.txt .
RUN pip install --no-binary=cffi -r requirements.txt -t /package

FROM public.ecr.aws/sam/build-python3.13 AS final
WORKDIR /var/task
COPY --from=builder /package /var/task
COPY bot_trading.py /var/task
