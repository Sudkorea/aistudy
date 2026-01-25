## 1. Albumentations의 표준 인터페이스

Albumentations 라이브러리는 이미지 증강(augmentation)을 위한 강력한 도구다. 이 라이브러리의 핵심 특징 중 하나가 바로 표준화된 인터페이스이다.

### 1.1 반환 형식

- Albumentations의 변환 메서드는 항상 **딕셔너리**를 반환함.
- 이 딕셔너리에는 변환된 이미지 데이터와 함께 다른 메타데이터가 포함될 수 있음.

### 1.2 딕셔너리 구조

일반적인 반환 딕셔너리의 구조는 다음과 같다:

```python
{
    'image': 변환된_이미지_데이터(NumPy 배열),
    'other_metadata': 기타_메타데이터
}
```

## 2. 변환된 이미지 사용하기

변환 후 이미지를 사용하려면, 딕셔너리에서 'image' 키로 접근하면 된다.

```python
transformed = transform_method(image=원본_이미지)
변환된_이미지 = transformed['image']
```

이렇게 하면 변환된 이미지 데이터(NumPy 배열)를 바로 얻을 수 있음

## 3. 장점

1. **일관성**: 모든 변환 메서드가 같은 형식으로 결과를 반환해서, 코드의 일관성 유지.
2. **편의성**: 딕셔너리 형태로 반환되니까, 필요한 정보만 쉽게 꺼내 쓸 수 있음
3. **확장성**: 나중에 추가 정보가 필요하면, 딕셔너리에 새로운 키-값 쌍을 추가하기 쉬움

## 4. 실제 사용 예

```python
def transform(transform_method):
    fig, axes = plt.subplots(1, 2, figsize=(15,7))
    axes[0].imshow(test_image)
    axes[1].imshow(transform_method(image=test_image_np)['image'])
    plt.show()

# 함수 작동 확인
transform1 = albumentations.HorizontalFlip(p=1)  # 입력 이미지의 좌우 반전
transform(transform1)
```

여기서 `transform_method(image=test_image_np)['image']`는 변환된 이미지 데이터를 바로 `imshow`에 전달해주는 것
## 정리

Albumentations 라이브러리는 dict 형태의 표준 인터페이스를 사용함. 변환한 이미지 정보와 함께 다른 메타데이터도 함께 반환해주는데, 우리는 이걸 그냥 딕셔너리에서 값 꺼내 쓰듯이 편하게 사용하면 됨