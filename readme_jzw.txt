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

####Training 
python train.py --model_name xxx --png --batch_size 1 

####训练自己的模型，ban掉auto——mask
python train.py --model_name xxx --png --batch_size 1 --disable_automasking

####在预训练的基础上训练自己的模型
python train.py --model_name xxx --png --batch_size 1 --disable_automasking  --load_weights_folder /home/robot/tmp/wights_save_2021_10_29/weights_19 --num_epochs 10
python train.py --model_name jzw1 --png --batch_size 1 --disable_automasking  --load_weights_folder /home/robot/tmp/xxx/models/weights_0 --num_epochs 4 --log_frequency 5

python train.py --model_name jzw1 --png --batch_size 1 --disable_automasking  --load_weights_folder /home/robot/tmp/jzw1/models/weights_1 --num_epochs 2 
参数
--model_name jzw1 --png --batch_size 12   --load_weights_folder /share/home/linandi/jzw/mono_640x192 --num_epochs 5 
--model_name nomask11_20 --png --batch_size 1 --disable_automasking   --load_weights_folder /share/home/linandi/jzw/mono_640x192 --num_epochs 5 
--model_name my1120automask --png --batch_size 1  --load_weights_folder /share/home/linandi/jzw/mono_640x192 --num_epochs 5
--model_name my11201641noautomask --png --batch_size 1  --disable_automasking --load_weights_folder /share/home/linandi/jzw/mono_640x192 --num_epochs 5


####评估pose
python evaluate_pose.py --eval_split odom_9 --load_weights_folder /home/robot/tmp/wights_save_2021_10_29/weights_19 --data_path kitti_odom/

####tensorboard查看
tensorboard --logdir=

####png转jpgs
find 09/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'

####查看cuda和cudnn
nvcc -V
cat /usr/local/cuda/version.txt

cat /usr/local/cuda/include/cudnn.h |grep CUDNN_MAJOR -A 2






###########docker###########
###image镜像###
docker pull imageName:TAG            # 从服务器拉取某个镜像到本地，TAG默认为latest
docker tag imageName1:TAG imageName2:TAG     # 将imageName1重名为imageimageName2
docker images                        # 列出本机所有image文件
docker rmi imageName:TAG             # 删除某个image，TAG默认为latest

###container容器####

docker container ls                  # 列出正在运行的容器
docker ps                            # 列出正在运行的容器
docker container ls --all            # 列出所有容器
docker ps -a                         # 列出所有容器
docker run [OPTIONS] imageName:TAG   # 创建一个新的容器，OPTIONS的值可查阅docker文档，例如：-it表示以交互模式运行容器
docker run --name ubuntu1604 -it ubuntu:16.04    # 创建一个以交互模式运行的ubuntu容器，名为“ubuntu1604”
docker kill containerID              # 传入容器ID终止容器运行
docker rm containerID1 containerID2 [...]      # 传入容器ID，删除一个或多个对应容器文件


###使用####
sudo docker run -itd ubuntu:18.04
sudo docker exec -it b130029f94e0 bash

docker中conda install pytorch 慢:使用清华镜像https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/



#1 拉取镜像
docker pull pytorch/pytorch:nightly-runtime-cuda10.0-cudnn7
#2 查看已下载的镜像
docker images
#3 在一个容器中运行想要的镜像
启动时注意挂载work-master目录到docker-container。
nvidia-docker run --rm -it --name mytest -v ~/sh/work-master/:/workspace pytorch/pytorch:nightly-runtime-cuda10.0-cudnn7
nvidia-docker run --rm -it --name mytest -v ~/monodepth2/:/workspace pytorch/pytorch:nightly-runtime-cuda10.0-cudnn7
nvidia-docker run --rm -it --net host --shm-size 8G --name mytest -v ~/monodepth2/:/workspace pytorch/pytorch:nightly-runtime-cuda10.0-cudnn7

创建的容器没有网络 添加参数 --nethost 使其与主机共享网络
conda activate失败  先激活环境 source activate  然后ca  退出环境 source deactivate
--rm 表示在结束container运行后，立刻删除container。不然的话还需要手动删除
-it 表示以交互方式启动
--name myRgmp 表示给启动的container一个名字，且其名字为mytest
-v 将本地目录挂载到docker-container中，这里就是把我要跑的代码的master作为workspace
# 注意这里的  ~/sh/work-master/   是ubuntu系统中程序代码所在的绝对路径，后面跟着的:/workspace注意要记得加上，貌似是为了挂载用
# pytorch/pytorch 为需要的镜像
# nightly-runtime-cuda10.0-cudnn7 为指定的镜像标签
#这里还要注意，启动方式要么为nvidia-docker，要么为docker run --runtime=nvidia
#一般用上面的就可以实现相关的操作，进入到容器中，并且处于master所在目录下
# 还有一个问题是记得要先把相关的文件在本地拷贝好再启动镜像和容器
#








