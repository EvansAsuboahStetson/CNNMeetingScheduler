import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # Output: (16, 28, 96)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: (16, 14, 48)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Output: (32, 14, 48)
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Output: (32, 7, 24)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (16, 14, 48)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (1, 28, 96)
            nn.Sigmoid()
        )



    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def compile_model(self, optimizer='adam', lr=0.001):
        self.criterion = nn.MSELoss()
        if optimizer == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        else:
            self.optimizer = optim.SGD(self.parameters(), lr=lr)

    def train_model(self, train_loader, epochs, validation_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                self.optimizer.zero_grad()
                output = self(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            print(f'Epoch {epoch + 1}, Average Loss: {running_loss / len(train_loader)}')

            # Evaluate on validation data
            avg_loss, avg_mae, avg_rmse, avg_mape = self.evaluate(validation_loader)
            print(f'Validation - Avg Loss: {avg_loss}, Avg MAE: {avg_mae}, Avg RMSE: {avg_rmse}, Avg MAPE: {avg_mape}')


    def evaluate(self, validation_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define device in the correct scope
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # No need to track gradients during evaluation
            total_loss, total_mae, total_rmse, total_mape = 0, 0, 0, 0
            for data, target in validation_loader:
                data, target = data.to(device), target.to(device)  # Ensure data is on the correct device
                output = self(data)
                loss = self.criterion(output, target)
                mae = torch.mean(torch.abs(output - target))
                mse = torch.mean((output - target) ** 2)
                mape = torch.mean(torch.abs((output - target) / target)) * 100

                total_loss += loss.item()
                total_mae += mae.item()
                total_rmse += mse.sqrt().item()
                total_mape += mape.item()

            # Average the totals over all batches
            num_batches = len(validation_loader)
            avg_loss = total_loss / num_batches
            avg_mae = total_mae / num_batches
            avg_rmse = total_rmse / num_batches
            avg_mape = total_mape / num_batches

        return avg_loss, avg_mae, avg_rmse, avg_mape

    def predict(self, data_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()
        predictions = []

        with torch.no_grad():  # No need to track gradients during prediction
            for data in data_loader:
                data = data.to(device)
                output = self(data)
                predictions.append(output.cpu())  # Move the predictions to CPU and store

        return torch.cat(predictions)  # Concatenate list of tensors into a single tensor




