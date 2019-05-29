import numpy  as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt

from torch.autograd import Variable


'''
This exercise is kind of similar to the process of training a denoising autoencoder.
You have in your database a certain amount of movies rated by certain amount of users.
However when building the recomender system you would like that your system is able 
to predict the rating that a user will give to a movie he has not seen. So what we do 
to test this is that with our current data base we erase some of the ratings a certain user
has given a then feed that to the autoencoder. 

In this way it is like wanting to compress an image but instead of the original image we with
a image with random noise expecting that the network will recover it.
'''

class StackedAutoencoders(nn.Module):

    def __init__(self, nb_movies ):

        super(StackedAutoencoders, self).__init__()

        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.activation(self.fc2(out))
        out = self.activation(self.fc3(out))
        out = self.fc4(out)
        return out

def save_checkpoint(optimizer, model, epoch, trLossH, valLossH, valAccH, filename):
    checkpoint_dict = {'optimizer': optimizer.state_dict(),
                       'model': model.state_dict(),
                       'epoch': epoch,
                       'trLossH': trLossH,
                       'valLossH': valLossH,
                       'valAccH': valAccH}

    torch.save(checkpoint_dict, filename)


def load_checkpoint(optimizer, model, filename):
    checkpoint_dict = torch.load(filename, map_location=DEVICE)
    epoch = checkpoint_dict['epoch']
    trLossH = checkpoint_dict['trLossH']
    valLossH = checkpoint_dict['valLossH']
    valAccH = checkpoint_dict['valAccH']

    model.load_state_dict(checkpoint_dict['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])

    return epoch, trLossH, valLossH, valAccH



MODEL = 1
DEVICE = 'cpu'

if __name__ == '__main__':

    #Import Dataset

    path = './Data/ml-1m/movies.dat'
    movies = pd.read_csv(path,sep='::',header=None,
                         engine = 'python', encoding = 'latin-1')

    path = './Data/ml-1m/users.dat'
    users = pd.read_csv(path,sep='::',header=None,
                         engine = 'python', encoding = 'latin-1')

    # path = './Data/ml-1m/ratings.dat'
    # ratings = pd.read_csv(path,sep='::',header=None,
    #                      engine = 'python', encoding = 'latin-1')

    #Prepare test set and training set
    path = './Data/ml-10k/u1.base'
    training_set = pd.read_csv(path,delimiter='\t')
    training_set = np.array(training_set, dtype = 'int')

    path = './Data/ml-10k/u1.test'
    test_set = pd.read_csv(path,delimiter='\t')
    test_set = np.array(test_set, dtype = 'int')

    #Getting the number of users and movies
    nb_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
    nb_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))

    #Turn test and train set into matrices and then into tensors.

    training_matrix = np.zeros((nb_users,nb_movies), dtype='float32')
    test_matrix = np.zeros((nb_users, nb_movies), dtype='float32')

    for idx in range(training_set.shape[0]):
        i,j, rating = training_set[idx, [0,1,2]]
        training_matrix[i-1,j-1] = rating

    for idx in range(test_set.shape[0]):
        i,j, rating = test_set[idx, [0,1,2]]
        test_matrix[i-1,j-1] = rating

    training_matrix = torch.from_numpy(training_matrix)
    test_matrix = torch.from_numpy(test_matrix)

    #Create model
    sae = StackedAutoencoders(nb_movies)
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(sae.parameters(), lr = 0.05, weight_decay = 0.3 )

    #TrainLogger
    valAccuracyHistory = []
    valLossHistory = []
    trLossHistory = []

    #Training model
    nb_epochs = 101
    first_epoch = 0
    load = True
    loadFile = 'Model1/classifier-f00-e{:03d}.pkl'.format(100)

    # Loading model parameters
    if load:
        first_epoch, trLossHistory, valLossHistory, valAccuracyHistory = load_checkpoint(optimizer, sae, loadFile)
        print("Loading model from epoch {:3d}".format(first_epoch))


    print("Start training model ")
    for epoch in range(first_epoch+1,first_epoch+ nb_epochs):
        train_loss = 0
        s = 0.0

        for id_user in range(0, nb_users):
            input = training_matrix[id_user,:].view(1,-1)
            target = input.clone()

            if torch.sum(target.data > 0) > 0:
                output = sae(input)
                target.requires_grad = False

                #Remove movies that were not rated
                remove = torch.from_numpy(np.ones((1,nb_movies), dtype='float32'))
                remove[target == 0] = 0
                clean_output = output * remove
                loss = criterion(clean_output, target)

                mean_corrector = nb_movies/float(torch.sum(target.data>0) + 1e-10)

                loss.backward()
                train_loss += np.sqrt(loss.item()*mean_corrector)
                s += 1.0

                optimizer.step()

        trLossHistory.append(train_loss/s)
        print("Epoch: {:03d}".format(epoch))
        print("Train Set Loss: {:.6f}".format(train_loss/s))

        # Test the model
        test_loss = 0
        s = 0.0

        for id_user in range(nb_users):
            input = training_matrix[id_user, :].view(1, -1)
            target = test_matrix[id_user, :].view(1, -1)

            if torch.sum(target.data > 0) > 0:
                output = sae(input)
                target.requires_grad = False

                # Remove movies that were not rated
                remove = torch.from_numpy(np.ones((1, nb_movies), dtype='float32'))
                remove[target == 0] = 0
                clean_output = output * remove
                loss = criterion(clean_output, target)

                mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10)

                test_loss += np.sqrt(loss.item() * mean_corrector)
                s += 1.0

        print("Test set Loss: {:.6f} \n".format(test_loss / s))
        valLossHistory.append(test_loss / s)

        #Save every 50 Epochs
        if epoch % 25 == 0 :
            checkpoint_filename = 'Model{:d}/classifier-f{:02d}-e{:03d}.pkl'.format(MODEL, 0, epoch)
            save_checkpoint(optimizer, sae, epoch, trLossHistory, valLossHistory, valAccuracyHistory,
                            checkpoint_filename)

            checkpoint_txt = 'Model{:d}/classifier-f{:02d}-e{:03d}-acc{:0.5f}.txt'.format(MODEL, 0, epoch,
                                                                                          min(valLossHistory))
            with open(checkpoint_txt, 'w') as f:
                pass


    #Plot performance
    fig, ax  = plt.subplots(2,1, sharex=True)

    ax[0].plot(trLossHistory, label ="Train Loss history" )
    ax[1].plot(valLossHistory, label ="Val Loss history" )

    print("Max accuracy in fold {:d}: {:.4f}".format(0,min(valLossHistory)))


    ax[0].set_title = "Accuracy vs Epoch for every fold."
    ax[0].set_ylabel("Accuracy")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()

    plt.show()