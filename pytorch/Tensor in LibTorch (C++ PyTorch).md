### n-차원 tensor 생성
```cpp
#include <torch/torch.h>

// 0-차원 (스칼라)
auto tensor_0d = torch::tensor(3.14);

// 1-차원
auto tensor_1d = torch::tensor({1, 2, 3, 4});

// 2-차원
auto tensor_2d = torch::tensor({{1, 2}, {3, 4}});

// 3-차원
auto tensor_3d = torch::tensor({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
```

### Tensor 데이터 타입
```cpp
// 정수형
auto int8_tensor = torch::tensor({1, 2, 3}, torch::kInt8);
auto uint8_tensor = torch::tensor({1, 2, 3}, torch::kUInt8);
auto int16_tensor = torch::tensor({1, 2, 3}, torch::kInt16);
auto int32_tensor = torch::tensor({1, 2, 3}, torch::kInt32);
auto int64_tensor = torch::tensor({1, 2, 3}, torch::kInt64);

// 실수형
auto float16_tensor = torch::tensor({1.0, 2.0, 3.0}, torch::kFloat16);
auto float32_tensor = torch::tensor({1.0, 2.0, 3.0}, torch::kFloat32);
auto float64_tensor = torch::tensor({1.0, 2.0, 3.0}, torch::kFloat64);

// 불리언
auto bool_tensor = torch::tensor({true, false, true}, torch::kBool);

// 복소수
auto complex64_tensor = torch::tensor({{1, 2}, {3, 4}}, torch::kComplexFloat);
auto complex128_tensor = torch::tensor({{1, 2}, {3, 4}}, torch::kComplexDouble);
```

### 주요 Tensor 생성 함수
```cpp
// 0으로 채워진 텐서
auto zeros = torch::zeros({3, 4});

// 1로 채워진 텐서
auto ones = torch::ones({2, 3});

// 균일 분포 랜덤 텐서
auto rand = torch::rand({3, 3});

// 정규 분포 랜덤 텐서
auto randn = torch::randn({2, 4});

// 순차적 값을 가진 텐서
auto arange = torch::arange(0, 10, 2);

// 선형 간격 값을 가진 텐서
auto linspace = torch::linspace(0, 1, 5);

// 단위 행렬
auto eye = torch::eye(3);

// 초기화되지 않은 텐서
auto empty = torch::empty({2, 3});

// 특정 값으로 채워진 텐서
auto full = torch::full({2, 2}, 3.14);
```

### Tensor 조작
```cpp
auto tensor = torch::randn({3, 4, 5});

// 형태 변경
auto reshaped = tensor.reshape({4, 3, 5});
auto viewed = tensor.view({60});

// 차원 추가/제거
auto squeezed = tensor.squeeze();
auto unsqueezed = tensor.unsqueeze(0);

// 전치
auto transposed = tensor.transpose(0, 1);

// 차원 순서 변경
auto permuted = tensor.permute({2, 0, 1});
```

### Tensor 연산
```cpp
auto a = torch::randn({2, 3});
auto b = torch::randn({2, 3});

auto sum = a + b;
auto diff = a - b;
auto prod = a * b;
auto div = a / b;
auto mat_mul = torch::matmul(a, b.transpose(0, 1));
auto pow = torch::pow(a, 2);
```

### GPU 사용
```cpp
if (torch::cuda::is_available()) {
    auto cuda_tensor = torch::randn({3, 3}, torch::kCUDA);
    std::cout << "Device: " << cuda_tensor.device() << std::endl;
}
```

### 자동 미분
```cpp
auto x = torch::randn({3, 3}, torch::requires_grad());
auto y = x * x + 3;
y.backward();
std::cout << "Gradient: " << x.grad() << std::endl;
```

### 파이썬 PyTorch와 C++ LibTorch 사이의 주요 차이점

1. 네임스페이스 사용:
   - C++에서는 `torch::` 네임스페이스를 사용함.
   예: `torch::tensor()`, `torch::randn()`

2. 자료형 선언:
   - C++에서는 `auto` 키워드를 사용하거나 명시적 자료형 선언이 필요함.
   예: `auto tensor = torch::randn({3, 3});` 또는 `torch::Tensor tensor = torch::randn({3, 3});`

3. 메서드 호출 문법:
   - 파이썬: `tensor.to('cuda')`
   - C++: `tensor.to(torch::kCUDA)`

4. 데이터 타입 지정:
   - 파이썬: `torch.tensor([1, 2, 3], dtype=torch.float32)`
   - C++: `torch::tensor({1, 2, 3}, torch::kFloat32)`

5. 디바이스 지정:
   - 파이썬: `torch.tensor([1, 2, 3], device='cuda')`
   - C++: `torch::tensor({1, 2, 3}, torch::kCUDA)`

6. 텐서 생성 시 중괄호 사용:
   - C++에서는 중괄호 `{}` 를 사용하여 초기 데이터나 shape를 지정함.
   예: `torch::tensor({1, 2, 3})`, `torch::zeros({3, 4})`

7. 인덱싱:
   - 파이썬: `tensor[0, 1]`
   - C++: `tensor.index({0, 1})`

#### LibTorch에서 음수 인덱스를 사용하는 방법

1. 단일 요소 접근:
   ```cpp
   torch::Tensor tensor = torch::tensor({1, 2, 3, 4, 5});
   auto last_element = tensor.index({-1});  // 마지막 요소 (5)
   auto second_last = tensor.index({-2});   // 뒤에서 두 번째 요소 (4)
   ```

2. 슬라이싱:
   ```cpp
   torch::Tensor tensor = torch::tensor({1, 2, 3, 4, 5});
   auto last_three = tensor.slice(0, -3);  // 마지막 3개 요소 (3, 4, 5)
   ```

3. 다차원 텐서에서의 사용:
   ```cpp
   torch::Tensor tensor = torch::tensor({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
   auto last_row = tensor.index({-1});  // 마지막 행 (7, 8, 9)
   auto last_column = tensor.index({"...", -1});  // 마지막 열 (3, 6, 9)
   ```

4. 고급 인덱싱:
   ```cpp
   torch::Tensor tensor = torch::tensor({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
   auto selected = tensor.index({torch::tensor({0, -1}), torch::tensor({-1, 0})});
   // 결과: tensor([3, 7])
   ```

주의할 점
- C++에서는 파이썬처럼 직접적인 대괄호 인덱싱 (`tensor[-1]`)을 사용할 수 없고, 대신 `index()` 메서드를 사용해야 함

- 슬라이싱에서 음수 인덱스를 사용할 때는 `slice()` 메서드를 사용함

- 다차원 인덱싱에서 `"..."`을 사용할 때는 문자열로 전달해야 함함

- 고급 인덱싱을 사용할 때는 인덱스 텐서를 생성하여 전달해야 함

### 슬라이싱:
   - 파이썬: `tensor[1:3, :]`
   - C++: `tensor.slice(0, 1, 3).slice(1, 0, tensor.size(1))`
### 텐서 속성 접근:
   - 파이썬: `tensor.shape`, `tensor.dtype`
   - C++: `tensor.sizes()`, `tensor.dtype()`

### 연산자 오버로딩:
- C++에서는 일부 연산자가 오버로딩되어 있지만, 때로는 함수를 사용해야 함.
예: `torch::matmul(a, b)` 대신 `a.matmul(b)` 사용

### require_grad 설정:
- 파이썬: `tensor.requires_grad_(True)`
- C++: `tensor.set_requires_grad(true)`

### CUDA 사용 가능 여부 확인:
- 파이썬: `torch.cuda.is_available()`
- C++: `torch::cuda::is_available()`    

### 에러 처리:
- C++에서는 예외 처리를 더 명시적으로 해야 할 수 있음.

### 메모리 관리:
- C++에서는 스마트 포인터나 명시적 메모리 해제를 고려해야 할 수 있음.

### Tensor Indexing 및 Slicing (C++ 버전)
#### 기본 Indexing

단일 요소에 접근할 때는 index() 메서드를 사용

```cpp
at::Tensor tensor = torch::randn({3, 4});
auto element = tensor.index({1, 2});  // 2행 3열의 요소
```

#### Slicing

slicing할 때는 slice() 함수와 함께 indexing을 사용

```cpp
at::Tensor tensor = torch::randn({4, 5});
auto sliced = tensor.index({torch::slice(1, 0, 3), torch::slice(0, 1, 4)});
```

이건 Python의 `tensor[1:4, 0:3]`와 같음

#### 고급 Indexing

불린 마스킹이나 팬시 인덱싱도 가능

```cpp
// 불린 마스킹
at::Tensor mask = tensor > 0;
auto positive = tensor.index({mask});

// 팬시 인덱싱
at::Tensor indices = torch::tensor({0, 2});
auto selected = tensor.index({indices});
```

#### 주의사항

1. C++에서는 Python만큼 직관적인 syntax가 없어서 좀 더 복잡해 보일 수 있음
2. index() 메서드는 항상 복사본을 반환하지 않아. 때로는 원본 데이터의 뷰를 반환할 수도 있음
3. 성능을 위해서 가능하면 연속적인 메모리 접근을 하는 게 좋음

#### 예시: 복잡한 Indexing

여러 조건을 조합해서 indexing하는 경우:

```cpp
at::Tensor tensor = torch::randn({5, 5});
auto complex_slice = tensor.index({
    torch::indexing::Slice(1, 4),
    torch::indexing::Slice(None, None, 2)
});
```

이건 Python의 `tensor[1:4, ::2]`와 같음

### 정리

| 메서드 | 주요 기능 | 메모리 연속성 필요 | 새 메모리 할당 | 사용 예시 |
|--------|-----------|---------------------|----------------|-----------|
| view() | 텐서 모양 변경 | 예 | 아니오 | `tensor.view({2, -1})` |
| flatten() | 지정 차원 평탄화 | 아니오 | 아니오 | `tensor.flatten(1)` |
| reshape() | 텐서 모양 변경 | 아니오 | 조건부 | `tensor.reshape({2, 8})` |
| transpose() | 두 차원 교환 | 아니오 | 아니오 | `tensor.transpose(0, 1)` |
| squeeze() | 크기 1인 차원 제거 | 아니오 | 아니오 | `tensor.squeeze()` |
| unsqueeze() | 새 차원 추가 | 아니오 | 아니오 | `tensor.unsqueeze(0)` |
| stack() | 텐서들을 새 차원으로 쌓기 | 아니오 | 예 | `at::stack({t1, t2})` |
| permute() | 여러 차원 재배열 | 아니오 | 아니오 | `tensor.permute({2, 0, 1})` |
| expand() | 텐서 크기 확장 | 아니오 | 아니오 | `tensor.expand({3, -1})` |
| repeat() | 텐서 반복 복제 | 아니오 | 예 | `tensor.repeat({2, 1})` |

1. C++에서는 대부분의 메서드에 중괄호 {}를 사용해 인자를 전달함. 이건std::initializer_list를 사용하기 때문

2. stack() 메서드는 at:: 네임스페이스의 함수로 사용됨. 다른 메서드들과는 조금 다르게 생겼으니 유의해야 함.

3. C++에서는 메모리 관리가 좀 더 명시적. 새 메모리 할당 여부를 더 신경 써야 할 수 있음

4. 성능 최적화를 위해 contiguous() 메서드를 자주 사용하게 될 것. 특히 view()나 다른 연산 전에 메모리 연속성을 확보하는 데 쓰임.