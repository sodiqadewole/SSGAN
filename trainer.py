from modules_ import *

def train(trainloader, config):
    # Create and initialize networks
    netg = NetG(config).to(config.device) 
    netd = NetD(config).to(config.device)
    netg.apply(weights_init)
    netd.apply(weights_init)
    
    optimizerD = optim.Adam(netd.parameters(), lr=config.lr, betas=(config.beta1, 0.9))
    optimizerG = optim.Adam(netg.parameters(), lr=config.lr, betas=(config.beta1, 0.9))

    G_losses = []
    D_losses = []
    netg.train()
    netd.train()
    real_label = torch.ones(size=(config.batch_size,), dtype=torch.float32, device=config.device)
    fake_label = torch.zeros(size=(config.batch_size,), dtype=torch.float32, device=config.device)
    
    for epoch in range(config.num_epochs):
        g_temp, d_temp = [], []
        # for each batch of the dataloader
        for i, data in enumerate(trainloader, 0):
            data, target = data[0].to(config.device, dtype=torch.float), data[1].to(config.device, dtype=torch.float)
            """
            This functions train the generator and the discriminator models
            """
            if config.resume:
                print("\nLoading pre-trained networks.")
                checkpoint = torch.load(model_path)
                netg.load_state_dict(checkpoint['netg_state_dict'])
                netd.load_state_dict(checkpoint['netd_state_dict'])
                optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
                optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
                print("\tDone.\n")

            # Forward-pass - netg
            fake, latent_i, latent_o = netg(data)

            # Forward pass - netd
            pred_real, feat_real = netd(data)
            pred_fake, feat_fake = netd(fake.detach())

            # Backward-pass - netg
            optimizerG.zero_grad()  # zero generator gradient
            # compute errors
            #netG_loss = compute_netg_errors(data, fake, latent_i, latent_o, feat_real, feat_fake, pred_fake, real_label, config)
            netG_loss = compute_netg_errors(data, fake, latent_i, latent_o, pred_fake, real_label, config)
            netG_loss.backward(retain_graph=True) # compute gradient
            optimizerG.step()  # back propagate error

            # Backward-pass - netd
            optimizerD.zero_grad()  # zero discriminator gradient
            # compute discriminator errors
            netD_loss = compute_netd_errors(pred_real, real_label, pred_fake, fake_label)
            netD_loss.backward(retain_graph=True) # compute gradient
            optimizerD.step()  # back propagate

            if i % 100 == 0:
                print(f"[{epoch}/{config.num_epochs}][{i}/{len(trainloader)}]\tLoss_D: {netD_loss.item():.6f}\tloss_G: {netG_loss.item():.6f}")

            # Save Losses for plotting later
            g_temp.append(netG_loss.item())
            d_temp.append(netD_loss.item())

        # Save Losses for plotting later
        G_losses.append(np.mean(g_temp))
        D_losses.append(np.mean(d_temp))

        os.makedirs(config.resume_path, exist_ok=True)
        model_path = os.path.join(config.resume_path, f'epoch_{epoch}.pth')

        torch.save({'epoch': epoch,
                    'netg_state_dict': netg.state_dict(),
                    'netd_state_dict': netd.state_dict(),
                    'optimizerG_state_dict': optimizerG.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict(),
                    #'loss': loss,
                    #...
                    }, model_path)

    import pandas as pd
    df_losses = pd.DataFrame()
    df_losses['G_losses'] = G_losses
    df_losses['D_losses'] = D_losses
    df_losses.to_csv('losses.csv', index=False)
    
    
def test_model(testloader=None, epoch=None, config=None):
    """This section test the discriminator on unseen dataset
        Args:
        model_path: path to saved model
        testloader: test loader dataset
    """
    import os
    from utils import get_norm_conf_matrix, get_conf_matrix
    
    model_path = os.path.join(config.resume_path, f'epoch_{epoch}.pth')
    #testloader = load_data(videos, mode='test')
    labels, predictions = [], []
    for i, test_data in enumerate(testloader):
        data, label = test_data[0].to(config.device, dtype=torch.float), test_data[1]
        netg, netd = load_model(epoch)
        netg.eval()
        netd.eval()
        netg.to(config.device)
        netd.to(config.device)
        # Forward-pass - netg
        #fake, latent_i, latent_o = netg(data)
        with torch.no_grad():
            pred_real, feat_real = netd(data)
            #pred_fake, feat_fake = netd(fake.detach())
        predictions.append(pred_real.cpu().detach().numpy())
        labels.append(label.squeeze())
        accuracy = get_norm_conf_matrix(labels, output)
    return accuracy, predictions, labels


def load_model(epoch=None):
    model_path = os.path.join(config.resume_path, f'epoch_{epoch}.pth')
    from networks import NetG, NetD 
    # instatiate models
    netg = NetG(config)
    netd = NetD(config)
    optimizerD = optim.Adam(netd.parameters(), lr=config.lr, betas=(config.beta1, 0.9))
    optimizerG = optim.Adam(netg.parameters(), lr=config.lr, betas=(config.beta1, 0.9))
    # load from check point
    checkpoint = torch.load(model_path)
    netg.load_state_dict(checkpoint['netg_state_dict'])
    netd.load_state_dict(checkpoint['netd_state_dict'])
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
    return netg, netd
