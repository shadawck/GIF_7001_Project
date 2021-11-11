# GIF_7001_Project

### Dev requirements

```sh
> pip install pipenv 
```

A la racine du projet run :

```sh
> pipenv sync
```

# GAN First order model test 

```sh
python3 demo.py --config config/vox-256.yaml --checkpoint data/vox-cpk.pth.tar --source_image first-order-model-demo/images/00.png --driving_video first-order-model-demo/videos/0.mp4 --result_video output.mp4 --cpu
```

Source images :

![Image source](./first-order-model/first-order-model-demo/images/00.png)


Source video :

https://user-images.githubusercontent.com/10743909/141328446-b0c16f80-5432-4baa-98c4-c46fdb46427a.mp4


OUTPUT : 

https://user-images.githubusercontent.com/10743909/141328526-ef330e9d-7d0a-4c83-8e2a-92f3475d8b2c.mp4

