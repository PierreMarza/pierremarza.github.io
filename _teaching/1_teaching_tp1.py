
## Dataset Loading
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Split train set into a train and validation sets
train_size = int(0.75*len(train_set))
valid_size = len(train_set) - train_size
train_set, valid_set = torch.utils.data.random_split(train_set, [train_size, valid_size])

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Ground-Truth classes in the CIFAR-10 dataset
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

## Dataloaders
# Define you batch size
batch_size = 4
# Training dataloader, we want to shuffle the samples between epochs
training_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
# Validation dataloader, no need to shuffle
valid_dataloader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0)
# Test dataloader, no need to shuffle
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

# Get a batch
dataiter = iter(training_dataloader)
images, labels = dataiter.next()
print('images: ', images.shape)
print('labels: ', labels.shape)

# Display batch -- Solution 1
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

# Display batch -- Solution 2
def process_img(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    return npimg

def imshow_batch(imgs, labels, classes):
    # Get batch_size
    bs = imgs.shape[0]
    # Create Matplotlib figure with batch_size sub_plots
    fig, axs = plt.subplots(1, bs)
    for i in range(bs):
        # Showing image
        axs[i].imshow(process_img(imgs[i]))
        # Removing axis legend
        axs[i].axis('off')
        # Adding the GT class of the image as a title of the subplot
        axs[i].title.set_text(classes[labels[i]])
    plt.show()

imshow_batch(images, labels, classes)



# Defining a Neural Network
import torch.nn.functional as F
# This is the base LeNet architecture you saw in the lecture, adapted to the our input and output dimensions
class LeNet(torch.nn.Module):
    def __init__(self):
        super (LeNet , self).__init__()
        # 3 input channels , 10 output channels ,
        # 5x5 filters , stride =1, no padding
        self.conv1 = torch.nn.Conv2d(3, 20, 5, 1, 0)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1, 0)
        self.fc1 = torch.nn.Linear(5*5*50, 500)
        self.fc2 = torch.nn.Linear(500, 10)

    def forward(self , x):
        x = F.relu(self.conv1(x))
        # Max pooling with a filter size of 2x2
        # and a stride of 2
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5*5*50)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = LeNet()
print('in: ', images.shape)
out = model(images)
print('out: ', out.shape)


# Feature Maps dimensions in CNN
def compute_output_shape_conv(input_shape=torch.Size([4, 3, 32, 32]), kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), n_out=20):
    assert len(input_shape) == 4, 'input shape should be (B, C, H, W)'
    assert len(kernel_size) == 2 and len(stride) == 2 and len(padding) == 2, 'all conv hyperparameters should be defined along both x and y axes' 

    I_x = input_shape[2]
    I_y = input_shape[3]

    out = []
    for i, I in enumerate([I_x, I_y]):
        O = 1 + (I - kernel_size[i] + 2*padding[i])/stride[i]
        out.append(int(O))
    
    return torch.Size([input_shape[0], n_out, out[0], out[1]])


class LeNet(torch.nn.Module):
    def __init__(self):
        super (LeNet , self).__init__()
        # 3 input channels , 10 output channels ,
        # 5x5 filters , stride =1, no padding
        self.conv1 = torch.nn.Conv2d(3, 20, 5, 1, 0)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1, 0)
        self.fc1 = torch.nn.Linear(5*5*50, 500)
        self.fc2 = torch.nn.Linear(500, 10)

    def forward(self , x):
        out_shape_conv1 = compute_output_shape_conv(x.shape, (5,5), (1,1), (0,0), 20)
        x = F.relu(self.conv1(x))
        assert x.shape == out_shape_conv1

        # Max pooling with a filter size of 2x2
        # and a stride of 2
        x = F.max_pool2d(x, 2, 2)

        out_shape_conv2 = compute_output_shape_conv(x.shape, (5,5), (1,1), (0,0), 50)
        x = F.relu(self.conv2(x))
        assert x.shape == out_shape_conv2

        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5*5*50)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

print(compute_output_shape_conv())

model = LeNet()
print('in: ', images.shape)
out = model(images)
print('out: ', out.shape)

# Loss function and optimizer
import torch.optim as optim

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Training loop
from tqdm import tqdm
epochs = 1
for epoch in range(epochs):
    logging_loss = 0.0
    for i, data in tqdm(enumerate(training_dataloader)):
        input, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        out = model(input)

        # Compute loss
        loss = criterion(out, labels)

        # Compute gradients
        loss.backward()

        # Backward pass - model update
        optimizer.step()

        logging_loss += loss.item()

        
        if i % 2000 == 1999:
            # Logging trainin loss
            logging_loss /= 2000
            print('Training loss epoch ', epoch, ' -- mini-batch ', i, ': ', logging_loss)
            logging_loss = 0.0
        
            # Model validation
            with torch.no_grad():
                logging_loss_val = 0.0
                for data_val in tqdm(valid_dataloader):
                    input_val, labels_val = data_val
                    out_val = model(input_val)
                    loss_val = criterion(out_val, labels_val)
                    logging_loss_val += loss_val.item()
                logging_loss_val /= len(valid_dataloader)
                print('Validation loss: ', logging_loss_val)  

# Save and load model
path = './le_net_cifar10.pth'

# Saving model
torch.save(model.state_dict(), path)

# Loading model
trained_model = LeNet()
trained_model.load_state_dict(torch.load(path))

# To use it for inference only, you can want to pass your model in eval mode
trained_model.eval()




    
