FROM tensorflow/tensorflow:1.12.0-gpu-py3
WORKDIR /app
# #指定系统编码.时区
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV LANG C.UTF-8

# #安装需要的库
COPY . .
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
RUN pip install -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com  -r requirements.txt
RUN apt-get clean && rm -r ~/.cache/pip
