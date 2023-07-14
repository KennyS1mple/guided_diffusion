# CACIOUS CODING
# Data     : 7/6/23  4:06 PM
# File name: test
# Desc     : 多卡训练测试 https://blog.csdn.net/baobei0112/article/details/122680749
#                       https://blog.csdn.net/ytusdc/article/details/122091284

import torch
import torchvision
import torch.backends.cudnn as cudnn
from torchvision import transforms
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import logging
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import numpy as np
import random
import comm


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False


def train(args, device):
    world_size = args.num_nodes * args.num_gpus
    distributed = args.num_nodes > 1  # 节点数大于1，则为分布式

    model = torchvision.models.resnet101(num_classes=10)
    model.conv1 = torch.nn.Conv1d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
    rank = args.node_rank
    model.to(device)

    init_seeds(2 + rank)  # 设定随即种子，保证每次生成固定的随机数

    cuda = device.type != 'cpu'
    # 单机多卡下的并行模型
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    if distributed:
        # SyncBatchNorm代替bn,需要DDP环境初始化后初始化
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # DDP model
        model = DDP(model, device_ids=[comm.get_local_rank()], output_device=comm.get_local_rank())
    total_batch_size = args.batch_size
    # batch_size per gpu
    batch_size = total_batch_size // world_size

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    data_set = torchvision.datasets.MNIST("./", train=True, transform=trans, target_transform=None, download=True)
    # datasets DDP mode
    sampler = torch.utils.data.distributed.DistributedSampler(data_set) if rank != -1 and distributed else None
    data_loader_train = torch.utils.data.DataLoader(dataset=data_set, batch_size=batch_size, shuffle=(sampler is None),
                                                    sampler=sampler)

    if rank in [-1, 0]:
        pass  # master node code, you can create test_datasets for val or test

    # loss opt
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    # net train
    for epoch in range(args.epochs):
        model.train()
        if rank != -1:
            # 设置当前的 epoch，为了让不同的节点之间保持同步
            data_loader_train.sampler.set_epoch(epoch)
        for i, data in enumerate(data_loader_train):
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            opt.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            print("loss: {}".format(loss.item()))
    # 在主节点上保存模型
    if args.node_rank in [-1, 0]:
        torch.save(model, "my_net.pth")


def main(args, device):
    ##################### 新添加代码 #########################
    world_size = args.num_nodes * args.num_gpus
    # 如果总节点数>1，则通过spawn启动各个进程
    if args.num_nodes > 1:
        if args.master_addr == "auto":
            assert args.num_nodes == 1, "dist_url=auto cannot work with distributed training."
            args.master_addr = '127.0.0.1'
        # spawn启动进程
        mp.spawn(
            _distributed_worker,  # 启动进程的函数
            nprocs=args.num_gpus,  # 每个节点启动进程的个数，即每个节点的gpu数量
            args=(train, world_size, device, args),  # 传入的参数
            daemon=False,
        )
    else:
        # 如果单机，则直接启动训练
        train(args, device)


def _distributed_worker(local_rank, train, world_size, device, args):
    # os.environ["CUDA_VISIBLE_DEVICES"] = 2,3,4,5  local_rank: 0,1,2,3
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."
    assert args.num_gpus <= torch.cuda.device_count()

    machine_rank = args.node_rank  # 节点编号
    num_gpus_per_machine = args.num_gpus  # 每个节点的gpu数量
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."
    global_rank = machine_rank * num_gpus_per_machine + local_rank  # 全局的rank编号

    init_method = 'tcp://{}:{}'.format(args.master_addr, args.master_port)  # 指定如何初始化进程组的URL
    try:
        # 采用nccl后端，推荐使用
        dist.init_process_group(
            backend="NCCL", init_method=init_method, world_size=world_size, rank=global_rank
        )
        print("init_process_group well done!")
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("Process group URL: {}".format(args.master_addr))
        raise e

    assert num_gpus_per_machine <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    assert comm._LOCAL_PROCESS_GROUP is None

    for i in range(args.num_nodes):
        ranks_on_i = list(range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine))
        pg = dist.new_group(ranks_on_i)  # 创建新组，具有所有进程的任意子集，它返回一个不透明的组句柄，作为所有集合体的“group”参数给出
        if i == machine_rank:
            comm._LOCAL_PROCESS_GROUP = pg  # 更新group祖
    train(args, device)  # 启动训练


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2, help="number of total batch_size")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument("--num_work", type=int, default=8, help="number of workers for dataloader")
    ########################### 新添加参数 ###########################
    parser.add_argument("--num_nodes", type=int, default=2, help="number of nodes")
    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpu per node")
    parser.add_argument("--node_rank", type=int, default=-1, help="the rank of this machine (unique per machine)")
    parser.add_argument("--master_addr", default="auto", help="url of master node")
    parser.add_argument("--master_port", default="29500", help="port of master node")
    ########################### 新添加参数 ###########################

    args = parser.parse_args()
    print("Command Line Args:", args)
    device = comm.select_device(args.device)
    main(args, device)
