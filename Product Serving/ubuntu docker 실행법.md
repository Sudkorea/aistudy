설치
```
wsl --install -d Ubuntu-22.04
```
실행
`wsl -d Ubuntu-22.04`

도커 설치
! 이미 윈도우 환경에 도커가 깔려있는 상황이라 에러뜸

```
docker pull ubuntu:latest

docker run -it --name emergency -v "/mnt/c/Users/findu/Desktop/중요하고 급한 폴더/assignments:/workspace" ubuntu:latest /bin/bash


```
sudo부터 깔아야하네
```
apt-get update

apt-get install sudo

sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### Docker 재시작 관련
```
docker stop (이름)

docker start (이름)
```

### sudo, apt update 관련
https://velog.io/@akfvh/sudoApt-vduqb7mk

