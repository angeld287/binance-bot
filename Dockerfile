FROM amazonlinux:2023

# Instala herramientas de build y dependencias
RUN yum update -y && \
    yum groupinstall -y "Development Tools" && \
    yum install -y openssl-devel libffi-devel bzip2-devel wget make

# Descarga y compila Python 3.13
RUN wget https://www.python.org/ftp/python/3.13.0/Python-3.13.0.tgz && \
    tar xzf Python-3.13.0.tgz && \
    cd Python-3.13.0 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall

# Confirma versión de Python instalada
RUN python3.13 --version

# Instala pip para Python 3.13
RUN python3.13 -m ensurepip && \
    python3.13 -m pip install --upgrade pip

# Copia tu requirements.txt
COPY requirements.txt .

# Instala requirements en /package, forzando compilación de cffi
RUN pip install --no-binary=cffi -r requirements.txt -t /package && \
    pip install patterns -t /package

# Confirma existencia de _cffi_backend.so
RUN find /package -name "_cffi_backend*.so"

# --- Copiar código manteniendo la MISMA estructura en /package ---
RUN mkdir -p /package/core /package/analysis /package/strategies /package/config
COPY src/core/        /package/core/
COPY src/analysis/    /package/analysis/
COPY src/strategies/  /package/strategies/
COPY src/config/      /package/config/

# (opcional en contenedor; Lambda no lo requiere pero ayuda en pruebas)
ENV PYTHONPATH="/package:${PYTHONPATH}"

# Setea el directorio de trabajo para que el handler core.bot_trading.handler resuelva imports
WORKDIR /package
