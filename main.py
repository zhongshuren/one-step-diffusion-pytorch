import torch
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from params import parser
from dataset.circle import Circle
from model import DiMLP
from loss import DiffusionLoss

args = parser.parse_args()
device = torch.device('cuda')
dataloader = DataLoader(Circle(num_samples=256000), batch_size=args.batch_size, shuffle=True)
model = DiMLP(num_labels=4).cuda()
diffusion_loss = DiffusionLoss(model, num_labels=4)
optimizer = Adam(model.parameters(), lr=args.lr)

if __name__ == '__main__':
    info = {'train_loss': '#.###'}
    for epoch in range(args.epochs):
        p_bar = tqdm(total=len(dataloader), postfix=info, )
        model.train()
        for i, (x, c) in enumerate(dataloader):
            x, c = x.cuda(), c.cuda()
            loss = diffusion_loss(x, c)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            p_bar.update(1)
            info['train_loss'] = '%.3f' % loss.item()
            p_bar.set_postfix(info, refresh=True)
        p_bar.close()

        num_eval = args.num_eval
        model.eval()
        with torch.no_grad():
            e = torch.randn(num_eval, 2).to(device)
            c = torch.randint(0, 4, (num_eval,)).to(device)
            x = model(e, c=c)
            x = x.cpu().numpy()
            plt.figure(figsize=(5, 5))
            plt.scatter(x[:, 0], x[:, 1], s=1)
            plt.xlim(-5, 5)
            plt.ylim(-5, 5)
            plt.show()
