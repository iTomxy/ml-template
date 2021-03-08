# singularity

类似 docker，这里记录一些常用操作。


# 拉镜像（image）

```shell
singularity -d build /home/tom/singularity-images/dl.simg \
    docker://tyloeng/dl:v1
```

对应 docker 之 `docker pull tyloeng/dl:v1`，其中：

* 第一个参数是拉下来的 image 的保存路径、名字，按需自改。

* `docker://` 之后就是要拉的 image 名，在 dockerhub 可以看到。

拉完就可以在 /home/tom/singularity-images/ 下看到 dl.simg。


# 启动后台容器（instance）

```shell
export SINGULARITY_TMPDIR=/home/tom/tmp
export TMPDIR=/home/tom/tmp

singularity instance start --nv --fakeroot \
    -B /home/tom:/home/tom,/home/tom/dataset:/home/dataset \
    /home/tom/singularity-images/dl.simg \
    tom-1
```

类似 docker 之 `docker run`，其中：

* `--nv` 启用 gpu 支持，类似 docker 之 `--gpus all`。

* `--fakeroot` 确保容器**内**有 root 权限。（然而好像需要在宿主机有什么权限才能加这个参数）
    - 加了这个参数，会产生 `rootfs-*` 目录，见下一条。

* 两行 `export` 更改 Temporary Folders，防止 /tmp 被撑爆。
    - 自行改去一个容量大的分区。
    - 加 `--fakeroot` 会在这个指定的 tmp dir 下生成形如 `rootfs-*` 的目录，stop 容器之后还在，要手动删。
    - 改挂在自己目录下（如 /home/tom/ 下的 tmp/），删的时候好认。

* `-B` 挂载目录，类似 docker 之 `-v`。多个挂载点用逗号 `,` 分隔。

* `tom-1` 是容器名（instance name），自己随便取，连接容器时要用。


# 查看后台容器

```shell
singularity instance list
```

可以看到已启动的容器，类似 docker 之 `docker ps`。


# 连接后台容器

```shell
singularity shell instance://tom-1
```

类似 docker 之 `docker exec`。 `tom-1` 就是创建时指定的 instance name。


# 退出容器

```shell
exit
```

在容器内操作，命令提示符应该是 `Singularity>`。

exit 之后可以重连。


# 关后台容器

```shell
singularity instance stop tom-1
```

关了好像就！没！了！**不**像 docker stop，用 `docker ps -a` 还可以看到、可以重新 start；而更像是 `docker stop` + `docker rm`。

所以没事别 stop，exit 就好。如果用了 `--fakeroot`，stop 之后记得删掉产生的 `rootfs-*` 目录。
