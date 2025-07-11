FROM amazonlinux:2023 AS builder
RUN yum groupinstall -y "Development Tools" && \
    yum install -y gcc libffi-devel openssl-devel python3 python3-pip make zip

COPY requirements.txt .
RUN pip install -r requirements.txt -t /package && \
    pip freeze && \
    pip show cffi && \
    pip show _cffi_backend && \
    find / -name "*_cffi_backend*.so" && \
    echo "===== LISTING cffi folder =====" && \
    ls -la /package/cffi && \
    echo "===== END OF LIST =====" && \
    python3 - <<'EOF'
import pkgutil, sys
m = pkgutil.find_loader('_cffi_backend')
print(m.get_filename() if m else 'not found')
sys.exit(0 if m else 1)
EOF

FROM public.ecr.aws/sam/build-python3.13 AS final
WORKDIR /var/task
COPY --from=builder /package /var/task
COPY bot_trading.py /var/task
