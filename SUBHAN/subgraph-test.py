import dgl
import torch
from sklearn.metrics import f1_score
from utils import EarlyStopping
import numpy as np
import sklearn.metrics as sm
from sklearn.metrics import accuracy_score, classification_report
from dgl.dataloading import MultiLayerFullNeighborSampler, NodeDataLoader,NeighborSampler


def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average="micro")
    macro_f1 = f1_score(labels, prediction, average="macro")

    matrix = sm.confusion_matrix(prediction, labels)
    #print(matrix)
    report = classification_report(prediction, labels)
    print(report)

    return accuracy, micro_f1, macro_f1


def evaluate(model,dataset):
    model.eval()
    total_loss = 0
    total_acc = 0
    count = 0
    prediction = []
    labels = []
    loss_fcn = torch.nn.CrossEntropyLoss()
    #loss_fcn = torch.nn.BCELoss()
    #test_list = [19,20,21,22,23]
    for i in range(120,158):
        g = dataset[i]
        '''
        得到input_nodes中哪些节点需要预测
        '''
        with torch.no_grad():
            features = g.nodes['flow'].data['feat']
            features = torch.tensor(features, dtype=torch.float)
            logits = model(g, features) 
        lab = g.nodes['flow'].data['label']
        loss = loss_fcn(logits,lab.long())
        _, indices = torch.max(logits, dim=1)
        pre = indices.long().cpu().numpy()
        acc =  (pre == lab.cpu().numpy()).sum() / len(pre)
        print(acc)
        total_loss += loss.item() * len(pre)
        total_acc += acc
        count += len(pre)
        '''
        计算混淆矩阵
        '''
        prediction = prediction + indices.numpy().tolist()
        labels = labels + lab.numpy().tolist()
    micro_f1 = f1_score(labels, prediction, average="micro")
    macro_f1 = f1_score(labels, prediction, average="macro")

    matrix = sm.confusion_matrix(prediction, labels)
    #print(matrix)
    report = classification_report(prediction, labels)
    print(report)
    return total_loss / count, total_acc / count, micro_f1, macro_f1

def main(args):
    #加载数据集
    dataset = dgl.data.CSVDataset('./portscan_origin')
    #加载模型
    num_classes = 2
    features = dataset[0].nodes['flow'].data['feat']
    if args["hetero"]:
        from han import HAN
        model = HAN(
            meta_paths=[["rsend", "send"], ["receive", "rreceive"], ["target", "rtarget"],["self-flow"]],
            in_size=features.shape[1],
            hidden_size=args["hidden_units"],
            out_size=num_classes,
            num_heads=args["num_heads"],
            dropout=args["dropout"],
        ).to(args["device"])    
    else:
        from model import HAN

        model = HAN(
            num_meta_paths=len(g),
            in_size=features.shape[1],
            hidden_size=args["hidden_units"],
            out_size=num_classes,
            num_heads=args["num_heads"],
            dropout=args["dropout"],
        ).to(args["device"])
        g = [graph.to(args["device"]) for graph in g]

    stopper = EarlyStopping(patience=args["patience"])
    loss_fcn = torch.nn.CrossEntropyLoss()#weight=torch.tensor([1.0,15.0])
    #loss_fcn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    for epoch in range(args["num_epochs"]):
        model.train()
        for i in range(120):
            g = dataset[i]
            g = g.to(args["device"])
            #print(g)
            #根据模型预测节点
            features = g.nodes['flow'].data['feat']
            features = torch.tensor(features, dtype=torch.float)
            labels = g.nodes['flow'].data['label']
            logits = model(g, features)
            #print(logits)
            #logits =  torch.max(logits, dim=1)
            loss = loss_fcn(logits,labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, indices = torch.max(logits, dim=1)
            prediction = indices.long().cpu().numpy()
            train_acc = (prediction == labels.cpu().numpy()).sum() / len(prediction)
            print(
                "Epoch {:05d} | Train Acc: {:.4f} | Train Loss: {:.4f}".format(
                    epoch, train_acc, loss.item()
                )
            )
        val_loss, val_acc, micro_f1, macro_f1 = evaluate(
            model, dataset
        )
        print(
            "Epoch {:05d} | Valid Acc: {:.4f} | Valid loss: {:.4f} | micro_f1: {:.4f} | macro_f1: {:.4f}".format(
                epoch, val_acc, val_loss, micro_f1, macro_f1
            )
        )


if __name__ == "__main__":
    import argparse

    from utils import setup

    parser = argparse.ArgumentParser("HAN")
    parser.add_argument("-s", "--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "-ld",
        "--log-dir",
        type=str,
        default="results",
        help="Dir for saving training results",
    )
    parser.add_argument(
        "--hetero",
        action="store_true",
        default=True,
        help="Use metapath coalescing with DGL's own dataset",
    )
    args = parser.parse_args().__dict__
    print(args)

    args = setup(args)

    main(args)