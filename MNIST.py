from fastai.vision.all import *
from fastbook import *
import pandas as pd
import plotext as plt


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
    
class Trainer():
    def __init__(self, model, train_data, valid_data, lr: float, epochs: int):
        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.lr = lr
        self.epochs = epochs
        self.loss = 0
        self.loss_hist = []
        self.acc_hist = []

    def update_params(self):
        for param in self.model.parameters():
            param.data -= self.lr*param.grad.data
            param.grad = None

    def train(self):
        for x,y in self.train_data:
            preds = self.model(x)
            self.loss = self.cross_entropy_loss(preds, y)
            self.loss.backward()
            self.update_params()
    
    def train_loop(self):
        for i in range(self.epochs):
            self.train()
            self.acc_hist.append(self.accuracy()*100)
            self.loss_hist.append(self.loss)
            self.plot_losses()
            
    def softmax(self, preds):
        preds = preds-torch.max(preds)
        return torch.exp(preds)/torch.sum(torch.exp(preds), dim=1).unsqueeze(1)
    
    def cross_entropy_loss(self, preds, trgt):
        log_soft = torch.log(self.softmax(preds))
        one_hot = log_soft[range(len(log_soft)),trgt.flatten()]
        cel = -torch.sum(one_hot)
        return cel

    def validate_epoch(self):
        accs = [self.batch_accuracy(self.model(x),y) for x,y in self.valid_dl]
        return round(torch.stack(accs).mean().item(), 4)
        
    def batch_accuracy(self, x, y):
        preds = self.softmax(x)
        predicted_value = torch.argmax(preds, dim=1)
        trgts = y.flatten()
        bools = predicted_value == trgts
        accs = bools.to(torch.float).mean()
        return accs
    
    def accuracy(self):
        return torch.stack([self.batch_accuracy(self.softmax(self.model(x)), y) for x,y in self.valid_data]).mean()

    def plot_losses(self):
        plt.clt()
        plt.cld()

        plt.theme("pro")
        plt.plot_size(80,40)
        plt.ylim(0, self.loss_hist[-1]+10, "right")
        plt.ylim(min(self.acc_hist), 100, yside="left")
       
        plt.xlim(0, self.epochs)
        plt.xticks(range(len(self.loss_hist)))
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy", yside="left")
        plt.ylabel("Loss", yside="right")
        plt.title("Model Vitals")
        
        plt.text(f"LR: {self.lr}", self.epochs*.75, 100, color="red+")
        plt.text(f"Accuracy: {self.acc_hist[-1]:.2f}%", self.epochs*.75, 99, color="orange+")
        plt.text(f"Loss: {self.loss_hist[-1]:.2f}",self.epochs*.75, 99.5, color="cyan+")
        plt.plot(self.loss_hist, yside="right", label = "Loss", color="cyan+")
        plt.plot(self.acc_hist, yside="left", label="Accuracy", color="orange+")
        
        plt.sleep(0.001)
        plt.show()



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

    epochs = 100
    lr = 0.003
    model = SimpleNN(28*28, 30, 10)
    trainer = Trainer(model, training, validation, lr, epochs)
    trainer.train_loop()
        

if __name__ == "__main__":
    main()