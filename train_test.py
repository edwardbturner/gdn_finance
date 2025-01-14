import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam


def custom_loss(y_hat: torch.Tensor, y: torch.Tensor, gamma: float, delta: float) -> torch.Tensor:
    """
    Calculate our custom loss function.

    :param y_hat: torch.Tensor - Predicted values.
    :param y: torch.Tensor - Actual values.
    :param gamma: float - Scaling factor for the tanh component.
    :param delta: float - Weighting factor between the two loss components.
    :return: torch.Tensor - Computed loss.
    """
    loss = delta * torch.mean(-torch.tanh(gamma * y_hat) * y) + (1 - delta) * torch.mean((y_hat - y) ** 2)
    return loss


def pnl_function(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Calculate the profit and loss (PnL) function.

    :param y_hat: torch.Tensor - Predicted values.
    :param y: torch.Tensor - Actual values.
    :return: torch.Tensor - Computed profit and loss.
    """
    pnl = torch.mean(torch.sign(y_hat) * y)
    return pnl


def rolling_train_test(
    model,
    data,
    GDN_epochs: int,
    DDPM_epochs: int,
    gamma: float,
    delta: float,
    GDN_lr: float,
    DDPM_lr: float,
    DDPM_lb: int,
    training_lb: int,
    GCN: bool,
) -> tuple[list[list[float]], list[list[float]], list[float], list[float], np.ndarray]:
    """
    Perform rolling train and test on the model.

    :param model: The model to train and test.
    :param data: The dataset to use.
    :param GDN_epochs: int - Number of epochs for GDN training.
    :param DDPM_epochs: int - Number of epochs for DDPM training.
    :param gamma: float - Scaling factor for the tanh component in loss.
    :param delta: float - Weighting factor between the two loss components.
    :param GDN_lr: float - Learning rate for GDN.
    :param DDPM_lr: float - Learning rate for DDPM.
    :param DDPM_lb: int - Lookback period for DDPM.
    :param training_lb: int - Lookback period for training.
    :param GCN: bool - Boolean flag for using GCN.
    :return: Tuple containing lists of training losses, training PnLs, test losses, test PnLs, and a numpy array of
        test outputs.
    """
    train_losses: list[list[float]] = []
    train_pnls: list[list[float]] = []
    test_losses: list[float] = []
    test_pnls: list[float] = []
    test_ys: list[float] = []
    all_batches: list[list[np.ndarray]] = []

    T = data.snapshot_count

    for t in range(training_lb, T):
        print(f' ----- On train/test for day {t} ----- ')

        b = T if GCN else DDPM_lb + training_lb

        if t < b:
            model_passing_outputs = model_passing(
                model, data, GDN_epochs, gamma, delta, GDN_lr, training_lb, t, False, GCN
            )
            all_batches.append(model_passing_outputs[0])
            train_losses.append(model_passing_outputs[1])
            train_pnls.append(model_passing_outputs[2])
            test_losses.append(model_passing_outputs[3])
            test_pnls.append(model_passing_outputs[4])
            test_ys.append(model_passing_outputs[5])
            print(f'Current test pnl = {model_passing_outputs[4]}')

        else:
            model.train()
            previous_batches = torch.tensor(np.array(all_batches[-DDPM_lb:]))

            DDPMoptim = Adam(model._DDPM.parameters(), lr=DDPM_lr)
            mse = nn.MSELoss()
            num_steps = model.num_steps

            for epoch in range(DDPM_epochs):
                epoch_loss = 0.0

                for batch in previous_batches:
                    x0 = batch
                    n = len(x0)
                    epsilon = torch.randn_like(x0)
                    time = torch.randint(0, num_steps, (n,))
                    noisy_imgs = model._DDPM(x0, time, epsilon)
                    epsilon_theta = model._DDPM.backward(noisy_imgs, time.reshape(n, -1))
                    DDPM_loss = mse(epsilon_theta, epsilon)

                    DDPMoptim.zero_grad()
                    DDPM_loss.backward()
                    DDPMoptim.step()

                    epoch_loss += DDPM_loss.item()

                print(f"Loss at DDPM epoch {epoch + 1}: {epoch_loss/DDPM_lb:.6f}")

            model_passing_outputs = model_passing(
                model, data, GDN_epochs, gamma, delta, GDN_lr, training_lb, t, True, GCN
            )
            all_batches.append(model_passing_outputs[0])
            train_losses.append(model_passing_outputs[1])
            train_pnls.append(model_passing_outputs[2])
            test_losses.append(model_passing_outputs[3])
            test_pnls.append(model_passing_outputs[4])
            test_ys.append(model_passing_outputs[5])
            print(f'Current test pnl = {model_passing_outputs[4]:.6f}')

    return train_losses, train_pnls, test_losses, test_pnls, np.array(test_ys)


def model_passing(
    model,
    data,
    GDN_epochs: int,
    gamma: float,
    delta: float,
    GDN_lr: float,
    training_lb: int,
    t: int,
    use_DDRM: bool,
    GCN: bool,
) -> tuple[list[np.ndarray], list[float], list[float], float, float, tuple[np.ndarray, np.ndarray]]:
    """
    Pass the model through the data for training and testing.

    :param model: The model to use.
    :param data: The dataset to use.
    :param GDN_epochs: int - Number of epochs for GDN training.
    :param gamma: float - Scaling factor for the tanh component in loss.
    :param delta: float - Weighting factor between the two loss components.
    :param GDN_lr: float - Learning rate for GDN.
    :param training_lb: int - Lookback period for training.
    :param t: int - Current time step.
    :param use_DDRM: bool - Boolean flag for using DDRM.
    :param GCN: bool - Boolean flag for using GCN.
    :return: Tuple containing the current batch, training losses, training PnLs, test loss, test PnL, and test outputs.
    """
    current_batch: list[np.ndarray] = []
    train_t_losses: list[float] = []
    train_t_pnls: list[float] = []

    params = list(model._GCN.parameters()) + list(model._linear.parameters())
    optim = Adam(params, lr=GDN_lr)

    model.eval()
    x_data_list: list[np.ndarray] = []
    for i in range(t - training_lb, t + 1):
        m = data[i].x

        if use_DDRM:
            print(f'Sampling for day {i-t+training_lb+1}/{training_lb+1} which is day {i}')
            std_dev = m.std()

            # Standardize before denoising and scale back up post-DDRM
            m = m / std_dev
            x_data_list.append((model._DDRM(m, 0.05).detach().numpy()) * std_dev.item())
        else:
            x_data_list.append(m.detach().numpy())

    x_data = torch.tensor(np.array(x_data_list))

    model.train()
    for epoch in range(GDN_epochs):
        train_epoch_loss = 0.0
        train_pnl_loss = 0.0

        for i, t_ in enumerate(range(t - training_lb, t)):
            snapshot = data[t_]
            y_hat = model(x_data[i], snapshot.edge_index, snapshot.edge_type, snapshot.edge_attr, False)
            train_loss = custom_loss(y_hat, snapshot.y, gamma, delta)
            train_pnl_loss += pnl_function(y_hat, snapshot.y).item()

            optim.zero_grad()
            train_loss.backward()
            optim.step()

            train_epoch_loss += train_loss.item()

        print(f'On epoch {epoch+1} of day {t} training, loss = {train_epoch_loss/training_lb:.8f}')

        train_t_losses.append(train_epoch_loss / training_lb)
        train_t_pnls.append(train_pnl_loss / training_lb)

    model.eval()
    if not GCN:
        for i, t_ in enumerate(range(t - training_lb, t + 1)):
            snapshot = data[t_]
            x_0_hat = model(x_data[i], snapshot.edge_index, snapshot.edge_type, snapshot.edge_attr, True)
            x_0_hat = x_0_hat / x_0_hat.std()
            x_0_hat = x_0_hat[None, :]
            current_batch.append(x_0_hat.detach().numpy())

    snapshot = data[t]
    y_hat_t = model(x_data[-1], snapshot.edge_index, snapshot.edge_type, snapshot.edge_attr, False)
    test_t_loss = custom_loss(y_hat_t, snapshot.y, gamma, delta).item()
    test_t_pnl = pnl_function(y_hat_t, snapshot.y).item()

    output_y_hat = y_hat_t.detach().numpy()
    output_y = snapshot.y.detach().numpy()

    return current_batch, train_t_losses, train_t_pnls, test_t_loss, test_t_pnl, (output_y_hat, output_y)
