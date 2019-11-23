import torch 
from lstm import LSTM
from dataset import THUNewsDataset
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score
from eval import eval
def precision(target, output):
    output = torch.max(output, dim=1, keepdim=True)[1].tolist()
    # print(output)
    # print(target)
    return precision_score(target.cpu(), output, average='micro')

def train(model, optimizer, criterion, epoch, dataloader, test_dataloader):
    

    for e in range(epoch):
        model.train()
        epoch_loss = 0
        epoch_output = []
        epoch_target = []
        for batch_idx, (batch, target) in enumerate(dataloader):
            if torch.cuda.is_available():
                batch = batch.cuda()
                target = target.cuda()
            optimizer.zero_grad()
            output = model(batch)
            
            # print(output.shape)
            # print(target.shape)
            loss = criterion(output, target)
            loss.backward()

            # print(type(output))

            epoch_loss = epoch_loss + loss.item()
            epoch_output = epoch_output + torch.max(\
                    output, 1, keepdim=True)[1].tolist()
            epoch_target = epoch_target + target.tolist()

            optimizer.step()
            if (batch_idx + 1) % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    e, (batch_idx + 1) * len(batch), len(dataloader.dataset),
                    100. * (batch_idx + 1)/ len(dataloader), loss.item()
                ))
                # print(type(output))
                print('Batch Accuracy: {}'.format(precision(target, output)))
        print("\nTrain Epoch :{} Epoch Loss: {}, Accuracy {}\n".format(e, epoch_loss, \
                precision_score(epoch_target, epoch_output, average='micro')))
        
        torch.save(model, 'model/lstm.model.' + str(e))
        # test_model = torch.load('model/lstm.model.' + str(e))
        _, prec, _ = eval(model, test_dataloader)
        print("\nTrain Epoch :{}.  Test set Accuracy: {}".format(e, prec))

    torch.save(model.state_dict(), 'model/lstm.model')


if __name__ == "__main__":

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_set = THUNewsDataset('data/vocab.txt', 'data/train_set.csv', 256)
    test_set = THUNewsDataset('data/vocab.txt', 'data/test_set.csv', 256)
    # set parameters
    learning_rate = 1e-3
    print(len(train_set.vocab))
    params = {
        "vocab_size" : len(train_set.vocab),
        "embedding_dim" : 50,
        "lstm_hidden_dim": 50,
        "num_of_tags": 14, 
        "seq_max_length": 256
    }
    epoch = 5
    batch_size = 64


    model = LSTM(params)
    if torch.cuda.is_available():
        model = model.cuda()
    

    criterion = torch.nn.NLLLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    
    dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=1000, shuffle=False)
    train(model, optimizer, criterion, epoch, dataloader, test_dataloader)