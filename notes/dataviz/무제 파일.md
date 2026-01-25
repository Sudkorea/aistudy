| 메서드             | 설명                              | 예시                                                                       |
| --------------- | ------------------------------- | ------------------------------------------------------------------------ |
| `groupby()`     | 특정 컬럼을 기준으로 데이터를 그룹화            | `df.groupby('column')`                                                   |
| `agg()`         | 여러 집계 함수를 한 번에 적용               | `df.groupby('column').agg({'col1': 'sum', 'col2': 'mean'})`              |
| `transform()`   | 그룹별로 함수를 적용하고 원본과 같은 크기의 결과를 반환 | `df.groupby('column').transform('mean')`                                 |
| `filter()`      | 조건에 맞는 그룹만 선택                   | `df.groupby('column').filter(lambda x: len(x) > 2)`                      |
| `apply()`       | 각 그룹에 임의의 함수를 적용                | `df.groupby('column').apply(lambda x: x.max() - x.min())`                |
| `pivot_table()` | 데이터를 재구성하여 요약                   | `pd.pivot_table(df, values='value', index='column1', columns='column2')` |
| `melt()`        | 넓은 형태의 데이터를 긴 형태로 변환            | `pd.melt(df, id_vars=['A'], value_vars=['B', 'C'])`                      |
| `merge()`       | 두 데이터프레임을 결합                    | `pd.merge(df1, df2, on='key')`                                           |
| `concat()`      | 데이터프레임이나 시리즈를 연결                | `pd.concat([df1, df2])`                                                  |
| `resample()`    | 시계열 데이터 리샘플링                    | `df.resample('D').mean()`                                                |
| `rolling()`     | 이동 윈도우 계산                       | `df.rolling(window=7).mean()`                                            |
| `crosstab()`    | 두 변수의 빈도 테이블 생성                 | `pd.crosstab(df.column1, df.column2)`                                    |

`ax = plt.subplot()` 또는 `fig, ax = plt.subplots()`로 axes 객체를 생성한 후에는, 보통 ax 객체의 메서드를 직접 호출하여 그래프를 그리거나 수정함. 이렇게 하면 코드가 더 명확하고 직관적이며, 여러 그래프를 다룰 때 특히 유용함.

예시

```python
import matplotlib.pyplot as plt
import numpy as np

# 그래프 영역 생성
fig, ax = plt.subplots()

# 데이터 준비
x = np.linspace(0, 10, 100)
y = np.sin(x)

# ax 객체의 메서드를 사용하여 그래프 그리기
ax.plot(x, y)
ax.set_title('Sine Wave')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')

# 이미지 표시의 경우
# ax.imshow(image_data)

plt.show()
```

이 방식의 장점은:

1. 여러 서브플롯을 다룰 때 각 axes에 대해 개별적으로 작업할 수 있음
2. 코드의 가독성이 높아짐
3. 객체 지향적 접근 방식으로, matplotlib의 객체 모델을 더 잘 반영함
4. 그래프 커스터마이징이 더 쉽고 직관적임

`ax = ...` 형식으로 axes 객체에 직접 할당하는 방식은 드물게 사용되며, 특별한 경우나 복잡한 변환을 적용할 때 주로 쓰임