import torch
import numpy as np
import matplotlib.pyplot as plt
import tonic
import torchvision

from norse.torch import LIFParameters, LIFState
from norse.torch.module.lif import LIFCell, LIFRecurrentCell
# Notice the difference between "LIF" (leaky integrate-and-fire) and "LI" (leaky integrator)
from norse.torch import LICell, LIState

from typing import NamedTuple
from tqdm import tqdm, trange



class SNNState(NamedTuple):
    lif0 : LIFState
    readout : LIState


class SNN(torch.nn.Module):
    def __init__(self, input_features, hidden_features, output_features, tau_syn_inv, tau_mem_inv, record=False, dt=1e-6):
        super(SNN, self).__init__()
        self.l1 = LIFRecurrentCell(
            input_features,
            hidden_features,
            p=LIFParameters(alpha=100, 
                            v_th=torch.as_tensor(0.3),
                            tau_syn_inv=tau_syn_inv,
                            tau_mem_inv=tau_mem_inv,
                           ),
            dt=dt                     
        )
        self.input_features = input_features
        self.fc_out = torch.nn.Linear(hidden_features, output_features, bias=False)
        self.out = LICell(dt=dt)

        self.hidden_features = hidden_features
        self.output_features = output_features
        self.record = record

    def forward(self, x):
        seq_length, batch_size, _, _, _ = x.shape
        s1 = so = None
        voltages = []

        if self.record:
            self.recording = SNNState(
              LIFState(
                z = torch.zeros(seq_length, batch_size, self.hidden_features),
                v = torch.zeros(seq_length, batch_size, self.hidden_features),
                i = torch.zeros(seq_length, batch_size, self.hidden_features)
              ),
              LIState(
                v = torch.zeros(seq_length, batch_size, self.output_features),
                i = torch.zeros(seq_length, batch_size, self.output_features)
              )
            )

        for ts in range(seq_length):
            z = x[ts, :, :, :].view(-1, self.input_features)
            z, s1 = self.l1(z, s1)
            z = self.fc_out(z)
            vo, so = self.out(z, so)
            if self.record:
                self.recording.lif0.z[ts,:] = s1.z
                self.recording.lif0.v[ts,:] = s1.v
                self.recording.lif0.i[ts,:] = s1.i
                self.recording.readout.v[ts,:] = so.v
                self.recording.readout.i[ts,:] = so.i
            voltages += [vo]

        return torch.stack(voltages)

def decode(x):
    x, _ = torch.max(x, 0)
    log_p_y = torch.nn.functional.log_softmax(x, dim=1)
    return log_p_y

class Model(torch.nn.Module):
    def __init__(self, snn, decoder):
        super(Model, self).__init__()
        self.snn = snn
        self.decoder = decoder

    def forward(self, x):
        x = self.snn(x)
        log_p_y = self.decoder(x)
        return log_p_y

def get_data(batch_size):
    global trainset
    global testset
    
    transform = tonic.transforms.Compose(
        [
            tonic.transforms.ToSparseTensor(merge_polarities=True),
        ]
    )

    download = True
    trainset = tonic.datasets.POKERDVS(save_to='./data', download=download, train=True)
    testset = tonic.datasets.POKERDVS(save_to='./data', download=download, transform=transform, train=False)

    # reduce this number if you run out of GPU memory

    # add sparse transform to trainset, previously omitted because we wanted to look at raw events
    trainset.transform = transform

    train_loader = torch.utils.data.DataLoader(trainset,
                                            batch_size=batch_size,
                                            collate_fn=tonic.utils.pad_tensors,
                                            shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(testset,
                                            batch_size=batch_size,
                                            collate_fn=tonic.utils.pad_tensors,
                                            shuffle=False
    )

    return train_loader, test_loader


def get_model():
    global DEVICE

    LR = 0.002
    INPUT_FEATURES = np.product(trainset.sensor_size)
    HIDDEN_FEATURES = 100
    OUTPUT_FEATURES = len(trainset.classes)

    DEVICE = torch.device("cpu")

    model = Model(
        snn=SNN(
        input_features=INPUT_FEATURES,
        hidden_features=HIDDEN_FEATURES,
        output_features=OUTPUT_FEATURES,
        tau_syn_inv=torch.tensor(1/1e-4), 
        tau_mem_inv=torch.tensor(1/1e-4)
        ),
        decoder=decode
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    return model, optimizer


def train(model, device, train_loader, optimizer, epoch, max_epochs):
    model.train()
    losses = []

    for (data, target) in tqdm(train_loader, leave=False):
        data, target = data.to(device).to_dense().permute([1,0,2,3,4]), torch.LongTensor(target).to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    mean_loss = np.mean(losses)
    return losses, mean_loss

def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device).to_dense().permute([1,0,2,3,4]), torch.LongTensor(target).to(device)
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100.0 * correct / len(test_loader.dataset)

    return test_loss, accuracy

def train_model():
    EPOCHS = 5

    train_loader, test_loader = get_data(4)
    model, optimizer = get_model()

    training_losses = []
    mean_losses = []
    test_losses = []
    accuracies = []

    torch.autograd.set_detect_anomaly(True)

    for epoch in trange(EPOCHS):
        training_loss, mean_loss = train(model, DEVICE, train_loader, optimizer, epoch, max_epochs=EPOCHS)
        test_loss, accuracy = test(model, DEVICE, test_loader, epoch)
        training_losses += training_loss
        mean_losses.append(mean_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)

        path = f"./data/norse_tutorial_model/model_epoch_{epoch}.pth"
        torch.save(model.state_dict(), path)

    print(f"final accuracy: {accuracies[-1]}")


if __name__ == '__main__':
    train_model()