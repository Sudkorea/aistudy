```
---------------------------------------------------------------------------

RuntimeError                              Traceback (most recent call last)

[<ipython-input-69-2adce3d98881>](https://localhost:8080/#) in <cell line: 1>()
----> 1 plt.imshow(add_random_noise(img_t, scale=25))

[<ipython-input-68-4621f0941389>](https://localhost:8080/#) in add_random_noise(img_t, scale)
      2   # TODO 7-1) img_t와 모양(shape)이 같은 표준정규분포 난수 Tensor를 생성해 주세요. | 변수명 random_noise
      3   # TODO 7-1
----> 4   random_noise = torch.randn_like(img_t)
      5       6   print("random_noise.mean: {:.4f}".format(random_noise.mean()))

RuntimeError: "normal_kernel_cpu" not implemented for 'Byte'
```
img_t가 int8이라 나타나는 에러로 보임. `img_t.float()`으로 바꾸니 잘 작동함.