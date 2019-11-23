import torch 
from lstm import LSTM
from dataset import THUNewsDataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

def eval(model, test_dataloader):
    model.eval()
    loss = 0
    all_output = []
    all_target = []
    for batch, target in test_dataloader:
        if torch.cuda.is_available():
            batch = batch.cuda()
            target = target.cuda()
        # print(batch.shape)
        output = model(batch)
        # print(output.shape)
        all_output = all_output + torch.max(output, 1, keepdim=True)[1].tolist()
        all_target = all_target + target.tolist()
    # print(all_output[:10])
    # print(all_target[:10])
    return f1_score(all_target, all_output, average="micro"), \
        precision_score(all_target, all_output, average="micro"),\
        recall_score(all_target, all_output, average="micro")

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    max_seq_length = 256
    # train_set = THUNewsDataset('data/vocab.txt', 'data/train_set.csv', 256)
    test_dataset = THUNewsDataset('data/vocab.txt', 'data/test_set.csv', max_seq_length)

    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    params = {
        "vocab_size" : len(test_dataset.vocab),
        "embedding_dim" : 50,
        "lstm_hidden_dim": 50,
        "num_of_tags": 14, 
        "seq_max_length": 256
    }
    model = LSTM(params)
    model.load_state_dict(torch.load('model/lstm.model'))
    model.cuda()
    print(eval(model, test_dataloader))
    print(eval(model, test_dataloader))