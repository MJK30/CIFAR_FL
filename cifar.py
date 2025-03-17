import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from torchvision.transforms import Compose, Normalize, ToTensor


class Net(nn.Module):
    """
    Defines and Intializes a CNN Module 
    """
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 3 inputs, 6 filters, 5x5 kernals - Feature Extraction
        self.bn1 = nn.BatchNorm2d(6) # batch normalization
        self.pool = nn.MaxPool2d(2,2) # convert the Feature Map into 2x2 kernals - Dimensional Reduction
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 inputs, 16 filters, 5x5 kernals
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16*5*5, 120) # converts the feature map into a 1D vector - Flattening 
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120,84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x: Tensor) -> Tensor:
        """Computes the Forward Pass
        Args:
            x (Tensor): CNN
        Returns:
            Tensor: x
        """
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)
        return x

def load_data(partition_id : int):
    """
    Loads the CIFAR Dataset and partitions it.
    Creates the transforms and applies it in batches.
    Creates dataloaders for efficient data loading and processing
    """
    fds = FederatedDataset(dataset= "cifar10", partitioners={"train": 10})
    partition = fds.load_partition(partition_id)
    
    # divide the data in a single silo into training and test dataset
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5,0.5,0.5) , (0.5,0.5,0.5))]
    )

    
    def apply_transforms(batch):
        # apply the transforms to the batch
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    
    return trainloader,testloader

def train(net:Net, trainloader: torch.utils.data.DataLoader, epochs:int, device: torch.device) -> None:
    """
    Develop the training function.
    Define loss function, optimizer to minimize loss.
    Train the CNN model
    Args:
        net (Net): CNN Model
        trainloader (torch.utils.data.DataLoader): Train Dataset for the silo
        epochs (int): Runs
        device (torch.device): Device
    """
    criterion = nn.CrossEntropyLoss() # loss function.
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # SGD to minimize loss function
    
    print(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")
    
    # setup the model to train
    net.to(device= device)
    net.train()
    
    # loop over the dataset mulitple times for training
    for epoch in range(epochs):
        running_loss = 0.0
        for i,data in enumerate(trainloader, 0):
            image, label = data["img"].to(device), data["label"].to(device)
            
            # zero the parameter gradient
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(image)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            
            # print the stats
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

def test(net:Net, testloader: torch.utils.data.DataLoader, device: torch.device) -> Tuple[float, float]:
    """
    Develops the test function.
    Develops loss function. No gradient storage required
    Tests the global testdata in CNN
    Args:
        net (Net): CNN
        testloader (torch.utils.data.DataLoader): Global test dataset
        device (torch.device): device
    Returns:
        Tuple[float, float]: loss, accuracy
    """
    # define the loss and metric
    criterion = nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    
    # setup the model to test
    net.to(device)
    net.eval()
    
    # no need to store gradients
    with torch.no_grad():
        for data in testloader:
            image, label = data["img"].to(device), data["label"].to(device)
            outputs = net(image)
            loss += criterion(outputs,label).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == label).sum().item()
        accuracy = correct/len(testloader.dataset)
        return loss, accuracy




def main():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("CIFAR DATA TRAINING USING FEDERATED LEARNING")
    trainloader, testloader = load_data(0)
    net = Net().to(DEVICE)
    net.eval()
    print("Start Training!")
    train(net= net, trainloader= trainloader, epochs= 5, device= DEVICE)
    print("Evaluate model")
    loss, accuracy = test(net= net, testloader= testloader, device= DEVICE)
    print("loss: ", loss)
    print("accuracy: ", accuracy)

if __name__ == "__main__":
    main()