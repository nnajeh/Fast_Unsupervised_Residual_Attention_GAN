
alpha = 1.0
crit = nn.MSELoss()

optimizer_E = torch.optim.Adam(E.parameters(), lr = 1e-4, betas=(0.9,0.999)) # lr_d = 4e-4

padding_epoch = len(str(n_epochs))
kappa = 1.0
e_losses = []


train_total_e_losses, val_total_e_losses = [],[]
min_valid_loss_e = np.inf



for e in range(epochs):
        e_running_loss =0.0
        losses = []
        E.train()
      
        for i, (x, _,_) in enumerate(train_dataloader,0):
            #x = x.to(device)
            x = x.to(device, dtype=torch.float)

            code = E(x)
            
            rec_image = pretrained_G(code)

            d_input = torch.cat((x, rec_image.detach()), dim=0)

            f_x  = pretrained_D.extract_features(x)
            f_gx  = pretrained_D.extract_features(rec_image.detach())
            
            loss = crit(rec_image, x) + alpha * crit(f_gx, f_x)

            optimizer_E.zero_grad()
            loss.backward()

            optimizer_E.step()
            
            losses.append(loss.item())
            e_running_loss += loss.item()

        
        print(e, np.mean(losses))
        
        train_total_e_losses.append(np.mean(losses))
        
  
        image_check(rec_image.cpu())
        save_image(d_input*0.5+0.5, './rec'+str(e)+'.bmp')

        
               

        epoch_len = len(str(e))
        print(f"[{e:>{epoch_len}}/{e:>{epoch_len}}] "
              f"[E_Train_Loss: {loss.item()}]"
             )
        
        if e%20 ==0 and e!=0:
            torch.save(E.state_dict(), f"./E-{e}.pth")
