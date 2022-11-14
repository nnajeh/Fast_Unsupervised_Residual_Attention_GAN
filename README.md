# Fast_Unsupervised_Residual_Attention_GAN


## Abstract

Recently, deep unsupervised learning methods based on Generative Adversarial Networks (GANs) have shown great potential for detecting anomalies. These last can appear both in global and local areas of an image. Consequently, ignoring these local information may lead to unreliable detection of anomalies. In this paper, we propose a residual GAN-based unsupervised learning approach capable of detecting anomalies at both image and pixel levels. Our method is applied for COVID-19 detection, it is based on the BigGAN model to ensure high-quality generated images, also it adds attention modules to capture spatial and channel-wise features to enhance salient regions and extract more detailed features. The proposed model is composed of three components: a generator, a discriminator, and an encoder. The encoder enables a fast mapping from images to the latent space, which facilitates the evaluation of unseen images. We evaluate the proposed method with by real-world benchmarking datasets and a public COVID-19 dataset and we illustrate the performance improvement at image and pixel levels.

![Full-Framework](https://user-images.githubusercontent.com/38373885/195213222-858ec475-d0d4-4a9f-ba4c-0a371ece6fe7.png)




This is a PyTorch/GPU implementation of the paper [Fast Unsupervised Residual Attention GAN for COVID-19 Detection]:
```
@inproceedings{nafti2022fast,
  title={Fast Unsupervised Residual Attention GAN for COVID-19 Detection},
  author={Nafti, Najeh and Besbes, Olfa and Ben Abdallah, Asma and Bedoui, Mohamed Hedi},
  booktitle={Conference on Computational Collective Intelligence Technologies and Applications},
  pages={360--372},
  year={2022},
  organization={Springer}
}
```
