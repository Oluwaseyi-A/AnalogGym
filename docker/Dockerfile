FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace/AnalogGym

RUN apt-get update && apt-get install -y \
    build-essential wget git python3 python3-pip \
    bison flex libreadline-dev \
    && rm -rf /var/lib/apt/lists/*

# Install ngspice 43
RUN cd /tmp \
    && wget -O ngspice-43.tar.gz "https://downloads.sourceforge.net/project/ngspice/ng-spice-rework/old-releases/43/ngspice-43.tar.gz" \
    && tar -xzf ngspice-43.tar.gz \
    && cd ngspice-43 \
    && ./configure --with-readline=yes --disable-debug \
    && make -j$(nproc) \
    && make install \
    && ldconfig

RUN pip3 install numpy scipy matplotlib gymnasium

CMD ["bash"]