import torch 

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(63,1000)
        self.bn1 = torch.nn.BatchNorm1d(1000)

        self.linear2 = torch.nn.Linear(1000,1000)
        self.bn2 = torch.nn.BatchNorm1d(1000)

        self.linear3 = torch.nn.Linear(1000,500)
        self.bn3 = torch.nn.BatchNorm1d(500)

        self.linear4 = torch.nn.Linear(500,200)
        self.bn4 = torch.nn.BatchNorm1d(200)

        self.linear5 = torch.nn.Linear(200,50)
        self.output = torch.nn.Linear(50,3)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self,x):
        out = self.relu(self.bn1(self.linear1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.linear2(out)))
        out = self.dropout(out)
        out = self.relu(self.bn3(self.linear3(out)))
        out = self.dropout(out)
        out = self.relu(self.bn4(self.linear4(out)))

        out = self.relu(self.linear5(out))
        out = self.output(out)
        return out

def test():
    model = Model()
    noise = torch.randn((20,63))
    out = model.forward(noise)
    print(out.shape)
