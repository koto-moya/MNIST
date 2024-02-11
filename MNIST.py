from fastai.vision.all import *
from fastbook import *
import pandas as pd


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size1, output_size):
        super(SimpleNN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.ReLU = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size1,output_size)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.ReLU(x)
        x = self.linear2(x)
        return x

def validate_epoch(model, valid_dl):
        accs = [batch_accuracy(model(x),y) for x,y in valid_dl]
        return round(torch.stack(accs).mean().item(), 4)
    
def batch_accuracy(x, y):
        preds = softmax(x)
        predicted_value = torch.argmax(preds, dim=1)
        trgts = y.flatten()
        bools = predicted_value == trgts
        acc = bools.to(torch.float).mean()
        return acc

def softmax(preds):
    preds = preds-torch.max(preds)
    return torch.exp(preds)/torch.sum(torch.exp(preds), dim=1).unsqueeze(1)

def cross_entropy_loss(preds, trgt):
        soft = softmax(preds)
        one_hot = torch.zeros(trgt.shape[0], soft.shape[1])
        for i in range(one_hot.size(0)):
            index = trgt[i, 0].item()
            one_hot[i, int(index)] = 1
        loss = -torch.sum(torch.log(soft)*one_hot)
        return loss

def stacker(df):
    image_list = [torch.tensor(df.iloc[img].values) for img in range(len(df))]
    stacked = torch.stack(image_list).float()/255
    return stacked

def loader(data, labels):
    zipped_data = list(zip(torch.cat(data).view(-1, 28*28),tensor(labels).unsqueeze(1)))
    return DataLoader(zipped_data, batch_size=256, shuffle=True)

def imgs(train, test):

    testing = [stacker(test[test["label"] == i].iloc[:, 1:785]) for i in range(10)]
    validation = [stacker(train[train["label"] == i].iloc[0:(round(len(train[train["label"] == i])*.2)+1),1:785].reset_index().drop(labels ="index", axis=1)) for i in range(10)]
    training = [stacker(train[train["label"] == i].iloc[round(len(train[train["label"] == i])*.2)+1:,1:785].reset_index().drop(labels ="index", axis=1)) for i in range(10)]
    testing_labels = sum([[i]*len(testing[i]) for i in range(10)],[])
    training_labels = sum([[i]*len(training[i]) for i in range(10)],[])
    validation_labels = sum([[i]*len(validation[i]) for i in range(10)],[])
    return loader(testing, testing_labels), loader(validation, validation_labels), loader(training, training_labels)

def main():
    train = pd.read_csv("mnist_train.csv")
    test = pd.read_csv("mnist_test.csv")
    testing, training, validation = imgs(train, test)

    epochs = 1000
    lr = 0.0001
    model = SimpleNN(28*28, 30, 10)

    #opt = SGD(model.parameters(), lr)
    previous_length = 0
    for i in range(epochs):
        for x,y in training:
            preds = model(x)
            loss = cross_entropy_loss(preds, y)
            loss.backward()
            for param in model.parameters():
                param.data -= lr*param.grad.data
                param.grad = None
        if loss < 1.5:
            break
        acc = torch.stack([batch_accuracy(softmax(model(x)), y) for x,y in validation]).mean()

        message = f"\rLoss: {loss:.4f} | model accuracy: {acc*100:.0f}%"    
        padding = ' ' * (previous_length - len(message))
        print(message + padding, end='', flush=True)
        previous_length = len(message)
        time.sleep(.01)

if __name__ == "__main__":
    main()