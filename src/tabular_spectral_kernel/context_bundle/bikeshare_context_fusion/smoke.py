"""Small CPU/GPU gate-training and PSD/KRR check; no external data needed."""
import argparse
import json
from pathlib import Path
import torch
import run as B


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(17)
    device = B.DEVICE
    x = torch.randn(36, 3, dtype=torch.float64, device=device)
    y = torch.sin(x[:, 0]) + 0.2 * x[:, 1]
    kernels = torch.stack([torch.exp(-torch.cdist(x, x) / width) for width in [0.7, 2.0]])
    gate = B.SpectralRule(torch.zeros(3, device=device), torch.ones(3, device=device), outputs=2, atoms=4).double()
    optimizer = torch.optim.Adam(gate.parameters(), lr=0.02)
    losses = []
    for _ in range(5):
        weights = gate(x[24:]).softmax(-1)
        gram = torch.einsum('qm,mij->qij', weights, kernels[:, :24, :24])
        cross = torch.einsum('qm,mqi->qi', weights, kernels[:, 24:, :24])
        prediction = B.solve(gram, cross, y[:24].expand(12, -1), 1e-3)
        loss = (prediction - y[24:]).square().mean()
        optimizer.zero_grad(); loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in gate.parameters())
        optimizer.step(); losses.append(float(loss.detach()))
    minimum = float(torch.linalg.eigvalsh(gram.detach()).min())
    assert minimum > -1e-10
    assert losses[-1] < losses[0]
    record = dict(device=device, training_losses=losses, minimum_gram_eigenvalue=minimum, finite_predictions=bool(torch.isfinite(prediction).all()))
    (args.output / 'smoke.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
