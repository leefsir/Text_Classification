配置docker_gpu镜像环境

背景：当gpu资源有限，而使用的用户较多，单一版本的cuda不能满足tensorflow不同版本的支持需求。此时可以借助nvidia_docker2容器中调用宿主机gpu，进而通过不同版本的cuda_tensorflow镜像来实现环境隔离，满足各自的使用需求

nvidia_docker2的安装请参考：https://quantum6.blog.csdn.net/article/details/86416600?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromBaidu-1.control&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromBaidu-1.control

配置docker启动项使得docker==nvidia-docker（docker命令直接等同于nvidia-docker):

```shell
# 安装nvidia-container-runtime，默认安装位置：/usr/bin/
sudo apt-get install nvidia-container-runtime
# 修改 /etc/docker/daemon.json，配置默认nvidia运行
sudo vim /etc/docker/daemon.json 
# 在daemon.json文件中添加如下内容，如下示意图
{
"default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
# 重启docker服务即可生效
Sudo systemctl daemon-reload
Sudo systemctl restart docker
 
```

在当前文件夹下（同目录下包含dockerfile和requirements文件）执行命令：

```shell
docker build -t tf12_gpu:keras_bert0.80 .
```

