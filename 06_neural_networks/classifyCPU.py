## Load and normalize CIFAR10


import torch, torchvision, os, time, math
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

batch_size = 4

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataiter = iter(trainloader)
images, labels = dataiter.next()

# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(" ".join("%5s" % classes[labels[j]] for j in range(batch_size)))

## Define convolutional neural network

import torch.nn as nn
import torch.nn.functional as F

times = []
accuracies = []
file = open("infoCPU.txt", "a")
for count in range(10, 500, 10):

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, count, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(count, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = x
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()

    ## Define a loss function and optimizer
    # cross entropy and SGD with momentum

    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return "%dm %ds" % (m, s)

    start = time.time()
    end = 0
    ## Train the network
    # loop over data tensor, feed inputs into network and optimize
    PATH = "./cifar_netCPU.pth"
    if os.path.exists(PATH):
        print("Model already trained")
    else:
        for epoch in range(2):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs, labels

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)  # forward
                loss = criterion(outputs, labels)  # loss function
                loss.backward()  # backward
                optimizer.step()  # optimize

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(
                        "[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000)
                    )
                    running_loss = 0.0
                    end = timeSince(start)
                    print("%s" % (end))

        # save trained model
        PATH = "./cifar_netCPU.pth"
        torch.save(net.state_dict(), PATH)

    ## Test network on test data
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # # print images
    # imshow(torchvision.utils.make_grid(images))
    # print("GroundTruth: ", " ".join("%5s" % classes[labels[j]] for j in range(4)))

    # # load saved model
    # net = Net()
    # net.load_state_dict(torch.load(PATH))

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print("Predicted: ", " ".join("%5s" % classes[predicted[j]] for j in range(4)))
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images, labels
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        "Accuracy of the network on the 10000 test images: %d %%"
        % (100 * correct / total)
    )

    file.write(
        repr(count) + "\t" + repr(end) + "\t" + repr(100 * correct / total) + "\n"
    )
    os.remove(PATH)
    # # check performance of each class
    # correct_pred = {classname: 0 for classname in classes}
    # total_pred = {classname: 0 for classname in classes}

    # labs = []
    # preds = []
    # # again no gradients needed
    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data
    #         images, labels = images, labels
    #         outputs = net(images)
    #         _, predictions = torch.max(outputs, 1)
    #         # collect the correct predictions for each class
    #         for label, prediction in zip(labels, predictions):
    #             labs.append(label)
    #             preds.append(prediction)
    #             if label == prediction:
    #                 correct_pred[classes[label]] += 1
    #             total_pred[classes[label]] += 1

    # plt.figure(1)
    # for classname, correct_count in correct_pred.items():
    #     accuracy = float(correct_count) / total_pred[classname]
    #     misclassification = 1 - accuracy
    #     plt.bar(classname, misclassification)
    #     print(f"Accuracy for class {classname} is: {accuracy*100} %")
    # plt.title("Misclassification rate for each class")
    # plt.xlabel("Class")
    # plt.ylabel("Misclassification rate")
    # plt.savefig("images/misclassificationCPU.eps")
    # plt.show()

    # conf_matrix = confusion_matrix(labs, preds)
    # fig = plt.figure(2)
    # ax = fig.add_subplot(111)
    # cax = ax.matshow(conf_matrix)
    # for i in range(conf_matrix.shape[0]):
    #     for j in range(conf_matrix.shape[1]):
    #         plt.text(
    #             x=j,
    #             y=i,
    #             s=conf_matrix[i, j],
    #             va="center",
    #             ha="center",
    #             size="medium",
    #             color="white",
    #         )
    # cbar = fig.colorbar(cax)
    # plt.title("Confusion matrix of image classification")
    # plt.xlabel("Prediction")
    # plt.ylabel("Actual")
    # classnames = list(correct_pred.keys())
    # ax.set_xticklabels([""] + classnames, rotation=45)
    # ax.set_yticklabels([""] + classnames)
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    # cbar.set_label("Number of guesses", rotation=270, labelpad=20)
    # plt.savefig("images/confusion_matrixCPU.eps")
# use RNN for sequential datasets
#
