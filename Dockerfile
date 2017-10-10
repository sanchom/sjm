FROM ubuntu:16.04

RUN apt-get update && apt install -y \
    g++ \
    cmake \
    scons \
    wget \
    python-pip \
    libopencv-dev \
    libboost-all-dev \
    libprotobuf-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    protobuf-compiler \
    libprotoc-dev \
    git && \
    cd /usr/src/gtest && \
    cmake -DBUILD_SHARED_LIBS=ON . && \
    make && \
    mv libg* /usr/lib/ && \
    pip install numpy protobuf

RUN wget http://www.vlfeat.org/download/vlfeat-0.9.20-bin.tar.gz && \
    tar -xvzf vlfeat-0.9.20-bin.tar.gz && \
    mkdir /usr/local/include/vl && mv vlfeat-0.9.20/vl/*.h /usr/local/include/vl && \
    mv vlfeat-0.9.20/bin/glnxa64/libvl.so /usr/local/lib/ && \
    rm -rf vlfeat*

RUN git clone https://github.com/mariusmuja/flann.git && \
    cd flann && git checkout 59616e791c9f99f43f62edf6a39cd4e0aaef33b5 && \
    touch doc/manual.pdf && \
    mkdir build && cd build && cmake -DBUILD_MATLAB_BINDINGS=OFF -DBUILD_PYTHON_BINDINGS=OFF .. && make install && cd && \
    rm -rf flann

RUN apt-get remove -y cmake git wget && \
    apt-get autoremove -y
