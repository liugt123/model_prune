import argparse
import numpy as np
import os
import time
import onnx
import torch
import torch.nn.functional as F
from onnxsim import simplify
from torchvision import transforms
from torchvision.datasets import CIFAR10

import torch_pruning as tp
from model import BasicBlock, ResNet18
from trt.trt_engine import TRT_Engine

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    type=str,
    required=True,
    choices=[
        "train",
        "prune",
        "test",
        "tensorrt",
    ],
)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--verbose", action="store_true", default=False)
parser.add_argument("--total_epochs", type=int, default=100)
parser.add_argument("--step_size", type=int, default=70)
parser.add_argument("--round", type=int, default=1)

args = parser.parse_args()


def get_dataloader():
    train_loader = torch.utils.data.DataLoader(
        CIFAR10(
            "./data",
            train=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            ),
            download=False,
        ),
        batch_size=args.batch_size,
        num_workers=2,
    )
    test_loader = torch.utils.data.DataLoader(
        CIFAR10(
            "./data",
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            ),
            download=False,
        ),
        batch_size=args.batch_size,
        num_workers=2,
    )
    return train_loader, test_loader


def eval(model, test_loader):
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img = img.to(device)
            out = model(img)
            pred = out.max(1)[1].detach().cpu().numpy()
            target = target.cpu().numpy()
            correct += (pred == target).sum()
            total += len(target)
    return correct / total


def train_model(model, train_loader, test_loader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, 0.1)
    model.to(device)

    best_acc = -1
    for epoch in range(args.total_epochs):
        model.train()
        for i, (img, target) in enumerate(train_loader):
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = F.cross_entropy(out, target)
            loss.backward()
            optimizer.step()
            if i % 10 == 0 and args.verbose:
                print(
                    "Epoch %d/%d, iter %d/%d, loss=%.4f"
                    % (epoch, args.total_epochs, i, len(train_loader), loss.item())
                )
        model.eval()
        acc = eval(model, test_loader)
        print("Epoch %d/%d, Acc=%.4f" % (epoch, args.total_epochs, acc))
        if best_acc < acc:
            torch.save(model, "./checkpoints/resnet18-round%d.pth" % (args.round))
            best_acc = acc
        scheduler.step()
    print("Best Acc=%.4f" % (best_acc))


def prune_model(model):
    model.cpu()
    DG = tp.DependencyGraph().build_dependency(model, torch.randn(1, 3, 32, 32))

    def prune_conv(conv, amount=0.2):
        strategy = tp.strategy.L1Strategy()
        pruning_index = strategy(conv.weight, amount=amount)
        plan = DG.get_pruning_plan(conv, tp.prune_conv, pruning_index)
        plan.exec()

    block_prune_probs = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3]
    blk_id = 0
    for m in model.modules():
        if isinstance(m, BasicBlock):
            prune_conv(m.conv1, block_prune_probs[blk_id])
            prune_conv(m.conv2, block_prune_probs[blk_id])
            blk_id += 1
    return model


def main():
    train_loader, test_loader = get_dataloader()
    if args.mode == "train":
        args.round = 0
        model = ResNet18(num_classes=10)
        train_model(model, train_loader, test_loader)
    elif args.mode == "prune":
        previous_ckpt = "./checkpoints/resnet18-round%d.pth" % (args.round - 1)
        print("Pruning round %d, load model from %s" % (args.round, previous_ckpt))
        model = torch.load(previous_ckpt)
        prune_model(model)
        print(model)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM" % (params / 1e6))
        train_model(model, train_loader, test_loader)
    elif args.mode == "test":
        ckpt = "./checkpoints/resnet18-round%d.pth" % (args.round)
        print("Load model from %s" % (ckpt))
        model = torch.load(ckpt)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM" % (params / 1e6))
        acc = eval(model, test_loader)
        print("Acc=%.4f\n" % (acc))
    elif args.mode == "tensorrt":
        ckpt = "./checkpoints/resnet18-round%d.pth" % (args.round)
        print("Load model from %s" % (ckpt))
        model = torch.load(ckpt)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM" % (params / 1e6))
        torch_in = torch.ones((1, 3, 32, 32)).cuda()
        torch.onnx.export(
            model,
            torch_in,
            "./checkpoints/model_onnx.onnx",
            verbose=False,
            opset_version=12,
        )
        onnx_model = onnx.load("./checkpoints/model_onnx.onnx")
        model_simp, check = simplify(onnx_model)
        onnx.save(model_simp, "./checkpoints/model_onnx.onnx")

        cmd = (
            "onnx2trt "
            + "./checkpoints/model_onnx.onnx"
            + " -o "
            + "./checkpoints/tensorrt_engine.engine"
            + " -b "
            + "1"
            + " -w "
            + str(1024 * 1024 * 1024)
            + " -d 32"
        )
        os.system(cmd)

        trt_model = TRT_Engine("./checkpoints/tensorrt_engine.engine", max_batch_size=1)
        num_iter = 2000
        total_time_list = []
        with torch.no_grad():
            for i in range(num_iter):
                start = time.time()
                trt_model(torch_in)
                total_time_list.append(time.time() - start)
            print(
                "total FPS -> avg:{}, max:{}, min:{}".format(
                    1 / (sum(total_time_list[100:]) / (num_iter - 100)),
                    1 / (max(total_time_list[100:])),
                    1 / (min(total_time_list[100:])),
                )
            )


if __name__ == "__main__":
    main()
