

pretrained_G = Generator().to(device)
pretrained_G.load_state_dict(torch.load("./G.pth"))
pretrained_G.eval()

pretrained_D = Discriminator().to(device)
pretrained_D.load_state_dict(torch.load("./D.pth"))
pretrained_D.eval()

pretrained_E = Encoder().to(device)
pretrained_E.load_state_dict(torch.load("./E.pth"))
pretrained_E.eval()

### Image-Level anomaly detection Evaluation
residual_list, discrimination_list,  = [], []
score_list = []
gt_list, img_list, gt_mask_list = [], [], []
alpha = 1.0
with torch.no_grad():
    with open("./train_score_image_level.csv", "w") as f:
               f.write("labels, anomaly_score,\n")
                    
    for i, (imgs, labels, _) in enumerate(test_dataloader):
        if torch.cuda.is_available():

                img_list.extend(imgs.cpu().detach().numpy())
                gt_list.extend(labels.cpu().detach().numpy())                   
        
                real_img = imgs.to(device)
                bs = imgs.size(0)

                fake_imgs = pretrained_G(pretrained_E(real_img))
                
                d_input = torch.cat((real_img, fake_imgs), dim=0)
                
                
                #extracted features
                real_img_feat = pretrained_D.extract_features(real_img)
                fake_img_feat = pretrained_D.extract_features(fake_imgs.detach())
                
  
                anomaly_score_img = MSE(fake_imgs, real_img) + alpha * MSE(fake_img_feat, real_img_feat)
          
                
               # residual_list.append(recon_distance.cpu())

                score_list.append(anomaly_score_img.cpu())
              
            
                with open("./train_score_image_level.csv", "a") as f:
                     f.write(f"{labels.item()},"f"{anomaly_score_img}\n")
                






fig, plots = plt.subplots(1, 2, figsize=(20, 10))

#plot rocauc for images
fig_img_rocauc = plots[0]

#plot rocauc for pixels
fig_pixel_rocauc = plots[1]

total_roc_auc = []
total_pixel_roc_auc = []



# calculate image-level ROC AUC score: residual loss
fpr, tpr, _ = roc_curve(gt_list, score_list)
roc_auc = roc_auc_score(gt_list, score_list)
total_roc_auc.append(roc_auc)
print('%s ROCAUC: %.3f' % ('ROC', roc_auc))

precision, recall, thresholds = precision_recall_curve(gt_list, score_list)
roc_auc = auc(fpr, tpr)
pr_auc =  auc(recall, precision)



# Plot ROC-AUC
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC-AUC")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend()
plt.savefig('./Image_ROC-AUC_Image_level--1.png')
plt.show()




plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:3f}")
plt.title("PR-AUC")
plt.xlabel("Recall")
plt.ylabel("Pecision")
plt.legend()
plt.savefig('./Image_PR-AUC_Image_level--1.png')
plt.show()
