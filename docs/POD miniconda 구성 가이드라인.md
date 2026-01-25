준비물 : 인터넷 되는 노트북 + 그 안에 Docker

## 0) 랩탑에서 준비 (온라인)

```bash
# 작업 폴더 만들기 (여긴 자기 컴퓨터 안에 쓸 디렉토리 아무거나 설정함)
mkdir -p ~/py312_build/out && cd ~/py312_build
```

```bash
# amd64 컨테이너를 백그라운드로 띄우기 (끝까지 이 컨테이너만 쓸 것)
docker run --platform=linux/amd64 --name make-py312 -d \
  -v "$PWD/out":/out \
  quay.io/pypa/manylinux2014_x86_64 \
  sleep infinity
```

```bash
# 컨테이너 안에 micromamba 설치 (정적 바이너리)
docker exec make-py312 bash -lc \
  'curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /usr/local/bin bin/micromamba --strip-components=1'
```

```bash
# micromamba 루트 지정 (이 변수는 각 exec마다 넣어줘야 함)
docker exec make-py312 bash -lc 'export MAMBA_ROOT_PREFIX=/opt/micromamba && micromamba --help | head -n 1'
```

``` shell
# pyarrow, pyhive 설치
docker exec make-py312 bash -lc \
'export MAMBA_ROOT_PREFIX=/opt/micromamba && \
 micromamba install -y -p /opt/py312 -c conda-forge \
   pyarrow pyhive thrift_sasl pure-sasl && \
 micromamba clean -a -y'
```

```bash
# 테스트 : pyarrow, hive 잘 작동하는지
docker exec make-py312 bash -lc \
'export MAMBA_ROOT_PREFIX=/opt/micromamba && \
 micromamba run -p /opt/py312 python - << "PY"
import pyarrow as pa
from pyhive import hive
try:
    import thrift_sasl, pure_sasl
    print("OK:", "pyarrow", pa.__version__, "| thrift_sasl", thrift_sasl.__version__)
except Exception as e:
    print("OK (no-sasl-needed):", "pyarrow", pa.__version__)
    print("Note:", e)
PY'

```

```bash
# Python 3.12 + ipykernel + conda-pack 포함된 환경을 "경로"로 생성 (이름 대신 절대경로가 안전)
docker exec make-py312 bash -lc \
  'export MAMBA_ROOT_PREFIX=/opt/micromamba && \
   micromamba create -y -p /opt/py312 -c conda-forge python=3.12 pip ipykernel conda-pack && \
   micromamba clean -a -y'
```

```bash
# 동작 확인
docker exec make-py312 bash -lc \
  'export MAMBA_ROOT_PREFIX=/opt/micromamba && \
   micromamba run -p /opt/py312 python -V'
```

```bash
# conda-pack으로 환경 포장 → /out/py312.tar.gz (호스트의 ./out 에 바로 생성됨)
docker exec make-py312 bash -lc \
'export MAMBA_ROOT_PREFIX=/opt/micromamba && \
 mkdir -p /out && \
 micromamba run -p /opt/py312 conda-pack \
   -p /opt/py312 -o /out/py312.tar.gz'

```

```bash
# 산출물 확인 (호스트)
ls -lh out/py312.tar.gz
```

이제 `~/py312_build/out/py312.tar.gz`(및 `.sha256`)를 USB에 담기

---

## 1) 오프라인 서버(CentOS 7, glibc 2.17, x86_64)에서



```bash
# dependencies 폴더로 복사되었다고 가정
cd home/zetta/use/dependencies
```


```bash
# 서버 안에 conda 설치하는 부분

# USB에서 파일 복사 후:
bash Miniconda3-latest-Linux-*.sh -b -p "$HOME/miniconda3"
# conda 활성화 스크립트만 로드(전역 설정 금지)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda config --set offline true
# 기본 경로 고정(권장)
conda config --add pkgs_dirs "$HOME/conda/pkgs"
conda config --add envs_dirs "$HOME/conda/envs"

```

```bash
# 풀 위치(실행 가능한 파티션) 생성
mkdir -p 'home/zetta/use/dependencies/conda/envs/py312'
```

```bash
# 압축 해제
tar -xzf py312.tar.gz -C "home/zetta/use/dependencies/conda/envs/py312"
```

```bash
# 경로 정리 (conda-pack이 제공) — 반드시 실행
"home/zetta/use/dependencies/conda/envs/py312/bin/conda-unpack"
```

```bash
# 기본 동작 확인
"home/zetta/use/dependencies/conda/envs/py312/bin/python" -V
ldd "home/zetta/use/dependencies/conda/envs/py312/bin/python" | head -n 5   # 'not found' 없어야 정상
```

```bash
# Jupyter 커널 등록(사용자 영역)
"home/zetta/use/dependencies/conda/envs/py312/bin/python" -m ipykernel install --user \
  --name py312 --display-name "Python 3.12 (conda-pack)"
```

```bash
# 커널 리스트 확인
jupyter kernelspec list
```

Jupyter 새로고침 한번 하면 **Python 3.12 (conda-pack)** 커널 잡힐것


pyarrow 3.8에서 안땡겨오게 하기
```bash
KDIR="$HOME/.local/share/jupyter/kernels/py312"
ENV="$HOME/conda/envs/py312hive"

python - <<PY
import json, os

env = os.environ['ENV']
kdir = os.environ['KDIR']
p = f"{kdir}/kernel.json"

with open(p) as f:
    j = json.load(f)

# 1) argv 수정 (-s 옵션 추가해서 사용자 site 차단)
argv = j.get("argv", [])
if len(argv) >= 1 and (len(argv) == 1 or argv[1] != "-s"):
    argv = [argv[0], "-s"] + argv[1:]
j["argv"] = argv

# 2) 커널 전용 환경변수
ld = [d for d in (f"{env}/lib", f"{env}/lib64") if os.path.isdir(d)]
j.setdefault("env", {}).update({
    "PATH": f"{env}/bin:/usr/bin:/bin",
    "LD_LIBRARY_PATH": ":".join(ld),
    "PYTHONNOUSERSITE": "1",
    "PYTHONPATH": "",
    "PYTHONHOME": "",
    "CONDA_PREFIX": env,
    "PIP_USER": "false"
})

with open(p, "w") as f:
    json.dump(j, f, indent=2)

print("patched:", p)
PY

```
