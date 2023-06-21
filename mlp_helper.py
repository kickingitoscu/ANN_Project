import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

def mlp_train(train_data,hidden_dims, epochs, batch_size, learning_rate, cuda, plots,feature_dims,class_dims,test_data = None):
    # --- YOUR CODE HERE ---
    b_size = batch_size
    t_data = train_data
    if test_data is not None:
        tt_data = test_data # Test Data Set
    h_dims = hidden_dims # Vector of hidden dimensions
    lr = learning_rate
    cuda = cuda

    # For M1 - Macbook Air
    if (cuda == True):
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = 'cpu'


    # Training Data Loader for training Model
    dl_train = DataLoader(t_data,batch_size=b_size,shuffle=True)
    if test_data is not None:
        dl_test = DataLoader(tt_data,batch_size=100,shuffle=False)


    modules = [] # List used to implement variable Model with n Hidden Dimensions
    feature_dim = feature_dims #train_data[0].size(1) # in Features Dimension
    class_dim = class_dims #torch.unique(train_data[1]).size(0) # Out Features Dimension


    for i in range(len(h_dims)):
        # Batchnorm1d was added though no significant improvement was found, also failed to get images to verify
        # results of model
        if(i == 0):
            # First layer is always in dimensions (28*28) and goes to dimension specified on h_dims
            modules.append(nn.Linear(in_features = feature_dim,out_features = h_dims[i],bias = True))
            # modules.append(nn.BatchNorm1d(h_dims[i]))
            modules.append(nn.LeakyReLU())
        else:
            # Consequent layers go from h_dims[i-1] to h_dims[i]
            modules.append(nn.Linear(in_features = h_dims[i-1],out_features = h_dims[i],bias = True))
            # modules.append(nn.BatchNorm1d(h_dims[i]))
            modules.append(nn.LeakyReLU())

    # Last linear layer of the model, goes from last dimension of h_dim vector to categories to be classified
    modules.append(nn.Linear(in_features = h_dims[len(h_dims)-1],out_features = class_dim,bias = True))
    modules.append(nn.Softmax(dim = 1))

    mlp_model = nn.Sequential(*modules)
    mlp_model = mlp_model.to(device=device)
    #print(mlp_model)

    # Loss function
    loss_func = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.SGD(mlp_model.parameters(),lr=lr)
    # Lists to obtain results
    epochs_i = []
    losses2 = []
    accuracy_test = []
    accuracy_train = []
    test_loss = []
    d = 0

    # Iteration over epochs
    for epoch in range(epochs):
        d += 1
        loss_batch = 0
        loss_batch_test = 0
        j = 0
        k = 0
        mlp_model.train()
        for batch in dl_train:
            j += 1
            x_batch, y_batch = batch[0].to(device),batch[1].to(device)
            y_batch = y_batch.to(torch.int64)

            preds = mlp_model(x_batch)
            loss = loss_func(preds,y_batch)
            optimizer.zero_grad()
            loss.backward() # get gradients / automatic differentiation
            optimizer.step()
            loss_batch = loss_batch + loss.item()

        loss_batch_avg = loss_batch/j
        epochs_i.append(d)
        losses2.append(loss_batch_avg)
        acctrain = mlp_accuracy(mlp_model,t_data)
        accuracy_train.append(acctrain.item())
        print(f'{d} Iteration - Loss: ',{loss_batch_avg})
        # Calculate the test set loss
        mlp_model.eval()
        if test_data is not None:
            for batchT in dl_test:
                k += 1
                with torch.no_grad():
                    x_batch_t,y_batch_t = batchT[0],batchT[1]
                    y_batch_t = y_batch_t.to(torch.int64)
                    test_preds = mlp_model(x_batch_t)
                    test_loss_value = loss_func(test_preds, y_batch_t)
                    loss_batch_test = loss_batch_test + test_loss_value
            loss_batch_test_avg = loss_batch_test/k
            test_loss.append(loss_batch_test_avg.item())
            acctest = mlp_accuracy(mlp_model,tt_data)
            accuracy_test.append(acctest.item())

        # Condtion to break iteration of model when reaches 0.05 threshold
        if d > 10:
            if loss_batch_avg < 0.05:
                break

    if test_data is not None:
        losses_t = [epochs_i,test_loss]
        accuracy_r_test = [epochs_i,accuracy_test]
        x_accuracy = accuracy_test.index(max(accuracy_r_test[1]))+1
        y_accuracy = max(accuracy_r_test[1])

    losses_r = [epochs_i,losses2]
    accuracy_r_train = [epochs_i,accuracy_train]
    # Plots
    if plots == True:
        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10, 5))
        ax1.plot(losses_r[0],losses_r[1],'o',ls = '-',label = 'Avg Training Loss')
        if test_data is not None:
            ax1.plot(losses_t[0],losses_t[1],'o',ls = '-',label = 'Avg Test Loss')
        ax1.legend(loc='best')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Average Loss')
        ax1.set_title('Average Loss vs Epochs')
        ax2.plot(accuracy_r_train[0],accuracy_r_train[1],'o',ls = '-',label = 'Training Accuracy')
        if test_data is not None:
            ax2.plot(accuracy_r_test[0],accuracy_r_test[1],'o',ls = '-',label = 'Test Accuracy')
        ax2.legend(loc='best')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy vs Epochs')
        if test_data is not None:
            return mlp_model,(losses2,test_loss),fig
        else:
            return mlp_model,losses2,fig

    else:
        return mlp_model,losses2


def mlp_accuracy(mlp_model,test_data):

    # Data loader to calculate accuracy of model
    dl_test = DataLoader(test_data,batch_size=100,shuffle=False)
    correct = 0
    total = 0

    for batch in dl_test:
        test_logit = mlp_model(batch[0])
        pred = test_logit.max(dim=1).indices
        labels = batch[1]
        total += labels.size(0)
        correct += (pred == labels).sum()
    accuracy = 100*correct/total
    return accuracy
