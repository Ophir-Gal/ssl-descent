# Extra files loaded (May need for compiling AMP)
#   1) cuda/10.2.89     3) mpfr/3.1.4       5) ppl/1.2          7) dejagnu/1.6      9) isl/0.16.1 
#   2) gmp/6.1.1        4) mpc/1.0.3        6) cloog/0.18.4     8) autogen/5.18.7  10) gcc/7.5.0


from matplotlib import pyplot as plt
import seaborn as sns
import torch
from torchvision import datasets, transforms
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.nn.parallel
import numpy as np
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
from sklearn.model_selection import train_test_split
import time
import random
from os import path as p

try:    
    from apex import amp, optimizers
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this code")

torch.backends.cudnn.benchmark = True


# Define some util functions

def alpha_weight(epoch, T1, T2, af):
    """ calculate value of alpha for unlabeled data depending on the current epoch
  
    returns: 
        - alpha: 

    params:
        - epoch: 
        - T1:
        - T2:
        - af:
    """
    if epoch < T1:
        return 0.0

    elif epoch > T2:
        return af

    else:
        return ((epoch-T1) / (T2-T1))*af
    # return 1


def evaluate(model, test_loader):
    """ evaluate the network and get loss and accuracy values

    returns: 
        - (test_accuracy, test_loss)

    params:
        - model:
        - test_loader:
    """
    model.eval()
    correct = 0 
    running_loss = 0

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            output = model(data)
            predicted = torch.max(output,1)[1]
            
            correct += (predicted == labels).sum()
            running_loss += loss_fn(output, labels).item()

    return (float(correct)/len(mnist_test))*100, (running_loss/len(test_loader))

def _draw_(N, results, name=None):
    fig = plt.figure(figsize=(10,7))
    plt.xlabel('Complexity',fontsize=12)
    plt.ylabel('Loss',fontsize=12)
    plt.grid()
    list1, list2 = zip(*results)
    sns.lineplot(x=N,y=list1)
    sns.lineplot(x=N,y=list2)
    plt.draw()
    path = './DoubleDescent/'
    fig.savefig(p.join(path,"{1}_fig.png".format(name)))
    print(len(N),len(results))


# *********************************************************************************************************
# Define your neural net model
# *********************************************************************************************************

# # Architecture from : https://github.com/peimengsui/semi_supervised_mnist
# The paper's model is 
# " Our classifier was a neural network (NN) with one hidden layer consisting of 100 hidden units. We
# used ReLU activation function for hidden units, and used softmax activation function for all the output units. "

class Net(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_classes):
        super(Net, self).__init__()
        self.input_size = n_inputs
        self.output_size = n_classes
        self.hidden_size = n_hidden
        self.H = n_hidden
        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.cuda() # move model to cuda

    def add_hidden_units(self, n_new=1):
        self.cpu() # temporarily move model to cpu
        
        self.hidden_size += n_new # update hidden size count

        # take a copy of the current weights stored in self.fcs
        current = [ix.weight.data for ix in [self.fc1, self.fc2]]

        # make the new weights in and out of hidden layer you are adding neurons to
        hl_input = torch.zeros([n_new, current[0].shape[1]])
        nn.init.xavier_uniform_(hl_input)#, gain=nn.init.calculate_gain('relu'))

        hl_output = torch.zeros([current[1].shape[0], n_new])
        nn.init.xavier_uniform_(hl_input)#, gain=nn.init.calculate_gain('relu'))

        # concatenate the old weights with the new weights
        new_wi = torch.cat([current[0], hl_input], dim=0)
        new_wo = torch.cat([current[1], hl_output], dim=1)

        # reset weight and grad variables to new size
        self.fc1 = nn.Linear(current[0].shape[1], self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, current[1].shape[0])

        # set the weight data to new values
        self.fc1.weight.data = new_wi.clone().detach().requires_grad_(True)
        self.fc2.weight.data = new_wo.clone().detach().requires_grad_(True)
        
        # move model back to cuda
        self.cuda()


    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc1(x))
        x = self.log_softmax(self.fc2(x))

        return x



def train_super(model, optimizer, train_loader, test_loader):
    """ train model in a standard supervised way
    
    returns: 
        - result: 

    params:
        - model:
        - optimizer:
        - train_loader:
        - test_loader:
    """
    EPOCHS = 300
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.train()

    for epoch in (range(EPOCHS + 1)):
        running_loss = 0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
            output = model(X_batch)
            labeled_loss = loss_fn(output, y_batch)
                    
            optimizer.zero_grad()
            with amp.scale_loss(labeled_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            # clip_grad_norm_(amp.master_params(optimizer), 1000)
            optimizer.step()
            train_loss = labeled_loss.item()
            running_loss += train_loss

        if epoch % 100 == 0:
            test_acc, test_loss = evaluate(model, test_loader)
            print('Epoch: {} : Test Acc : {:.5f} | Test Loss : {:.3f}  : Train Loss : {:.3f}'.format(epoch, test_acc, test_loss, train_loss))
            model.train()

    return running_loss/len(train_loader), test_loss


def train_semisuper(model, optimizer, train_loader, unlabeled_loader, test_loader):
    """ 
    
    returns: 
        - alpha: 

    params:
        - model:
        - train_loader:
        - unlabeled_loader:
        - test_loader:
    """
    alpha_log = []
    test_acc_log = []
    test_loss_log = []

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    EPOCHS = 100
    T1 = 100
    T2 = 700
    af = 3

    # train model for on labeled data
    # standard supervised training
    print("STARTING STANDARD SUPERVISED LEARNING")
    super_train_loss, super_test_loss = train_super(model, optimizer, train_loader, test_loader)
    # super_results.append((train_loss, test_loss))
    
    # Instead of using current epoch we use a "step" variable to calculate alpha_weight. This helps the model converge faster
    step = 100 
    print("STARTING SEMI-SUPERVISED LEARNING")
    model.train()
    for epoch in (range(EPOCHS + 1)):
        running_unlabeled_loss = 0
        running_labeled_loss = 0

        for batch_idx, x_unlabeled in enumerate(unlabeled_loader):
            
            # forward pass to get the pseudo labels
            x_unlabeled = x_unlabeled[0].to(device)
            model.eval()
            output_unlabeled = model(x_unlabeled)
            _, pseudo_labeled = torch.max(output_unlabeled, 1)
            model.train()
            
            # Now calculate the unlabeled loss using the pseudo label
            output = model(x_unlabeled)
            unlabeled_loss = alpha_weight(step, T1, T2, af) * loss_fn(output, pseudo_labeled)   

            running_unlabeled_loss += unlabeled_loss.item()
            
            # Backpropogate
            optimizer.zero_grad()

            with amp.scale_loss(unlabeled_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            
            optimizer.step()

            # # For every 50 batches train one epoch on labeled data 
            # if batch_idx % 50 == 0:
                
            # Normal training procedure
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.cuda()
            y_batch = y_batch.cuda()

            output = model(X_batch)
            labeled_loss = F.nll_loss(output, y_batch)

            running_labeled_loss += labeled_loss.item()
            optimizer.zero_grad()

            with amp.scale_loss(labeled_loss, optimizer, delay_unscale=True) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
        # Now we increment step by 1
        step += 1

        train_loss = running_labeled_loss/len(train_loader) + running_unlabeled_loss/len(unlabeled_loader)    
        test_acc, test_loss = evaluate(model, test_loader)
        if(epoch % 50 == 0):
            print('Epoch: {} : Alpha Weight : {:.5f} | Total Train loss :  {:.5f} | Test Acc : {:.5f} | Test Loss : {:.3f} '.format(epoch, alpha_weight(step, T1, T2, af),train_loss, test_acc, test_loss))
        model.train()

    return train_loss, test_loss #, super_train_loss, super_test_loss


# ## Double Descent Risk Curve for Neural Networks
# The goal of this experiment is to analyze the existence of double descent risk curve for different neural networks trained using pseudo-labeling method with different datasets.
# - desired output : training & test risk vs. model complexity (e.g. number of parameters)
# - make sure your model is defined in pytorch nn.Module subclass
# - make sure your dataset is loaded in DataLoader class
# - The interpolation threshold (under-parameterized vs. over-parameterized) is observed at n * K (number of classes * number of samples)

# Set hyperparameters for this experiment

# In[6]:


# number of model parameters, learning rate, batch size, etc.
N = [i for i in range(1, 51, 1)]
n_batch = 50
n_classes = 10
n_labeled = 50000
n_unlabeled = 10000
lr = (10000)**(-0.5) 

sample_train_size = 1800
sample_test_size = 200

# Define your dataset


# load MNIST train/test
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
path = './data'

# Sample from the original data
mnist_train = datasets.MNIST(root=path, train=True, download=False, transform=transform)
mnist_test = datasets.MNIST(root=path, train=False, download=False, transform=transform)


## 20% label noise to train set
## Comment out below to turn off label noise
num_samples = len(mnist_train.targets)
rands = np.random.choice(sample_train_size, sample_train_size//4, replace=False)
for rand in rands:
    tmp = mnist_train.targets[rand]
    mnist_train.targets[rand] = np.random.choice( list(range(0,tmp)) + list(range(tmp+1,10)) )
## Comment out above to turn off label noise

train_set, val_set = torch.utils.data.random_split(mnist_train, [n_labeled, n_unlabeled])

mnist_train = torch.utils.data.Subset(train_set, np.random.randint(low=0, high=n_labeled, size=sample_train_size))
val_set = torch.utils.data.Subset(train_set, np.random.randint(low=0, high=n_unlabeled, size=(2000 - sample_train_size)))
mnist_test = torch.utils.data.Subset(mnist_test, np.random.randint(low=0, high=10000, size=sample_test_size))


train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=n_batch, shuffle=True)
unlabeled_loader = torch.utils.data.DataLoader(val_set, batch_size=n_batch, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=n_batch, shuffle=True)

# Define your loss function

# In[9]:


def loss_fn(outputs, labels):
    return F.nll_loss(outputs, labels)

# Evaluates the model by adding a new hidden unit (alternatively returns new model if weight reuse is false)
def get_model(channel, height, width, n_params, n_classes, reuse_wt, prev_model=None):
    if(reuse_wt == False or prev_model == None):
        return Net(channel*height*width, n_params, n_classes)
    else:
        model = prev_model
        model.add_hidden_units()
        return model
    
# In[11]:


# logging training and test risk (list[tuple] : [(model complexity, train_risk, test_risk)])
super_results = []
semisuper_results = []
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# extract dimensions of your dataset as an input parameter to defining the model
channel, height, width = next(iter(train_loader))[0].shape[-3:]

# for each model complexity (roughly defined as number of model parameters)
WEIGHT_REUSE = False

for n_params in N:

    start_time = time.time()
    if(n_params == N[0]):
        model_params = n_params
    else:
        model_params = sum(p.numel() for p in model.parameters())


    if(model_params > n_classes* sample_train_size):
        WEIGHT_REUSE = False

    if(n_params == N[0] or WEIGHT_REUSE == False):
        model = Net(channel*height*width, n_params, n_classes)
    else:
        prev_model = model
        model = get_model(channel, height, width, n_params, n_classes, WEIGHT_REUSE, prev_model).cuda() # define model with the number of parameters

    print('GPUs Available : ', torch.cuda.device_count())
    print('Device Name(s) : ', torch.cuda.get_device_name(0))
    print("number of model parameters={}".format(model_params))

    # model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    # loss scale is dynamic or '1.0'; add min_loss_scale=1.0 if dynamic
    # opt_level is '01' or '02' or '03'
    model, optimizer = amp.initialize(model, optimizer,
                                    opt_level= 'O2',     
                                    loss_scale=1.0,
                                    master_weights=True,
                                    keep_batchnorm_fp32=True,
                                    min_loss_scale=1.0
                                    )

    # standard supervised training

    # train_loss, test_loss = train_super(model, optimizer, train_loader, test_loader)
    # super_results.append((train_loss, test_loss))

    # semi-supervised training using pseudo-labels
    semi_train_loss, semi_test_loss = train_semisuper(model, optimizer, train_loader, unlabeled_loader, test_loader)
    semisuper_results.append((semi_train_loss, semi_test_loss))
    
    print("\n---* Section %d takes %.3f seconds ---*\n" % (n_params, (time.time() - start_time)))

print('*'*100)
print(N)
print(semisuper_results)
print('*'*100)


# save/load model for later use

# path = './Net/saved_models/supervised_weights'
# test_acc, test_loss = evaluate(model, test_loader)
# print('Test Acc : {:.5f} | Test Loss : {:.3f} '.format(test_acc, test_loss))
# torch.save(net.state_dict(), path)
# net.load_state_dict(torch.load(path))
