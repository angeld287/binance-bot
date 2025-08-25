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

# Copia tu script (si lo deseas probar en container)
COPY src/core/bot_trading.py /package/
COPY src/core/exchange.py /package/
COPY src/core/logging_utils.py /package/
COPY src/analysis/pattern_detection.py /package/
COPY src/analysis/resistance_levels.py /package/
COPY src/analysis/support_levels.py /package/
COPY src/analysis/sr_levels.py /package/
COPY src/strategies /package/strategies

# Setea el directorio de trabajo
WORKDIR /package
