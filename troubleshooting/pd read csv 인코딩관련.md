## 문제
csv 파일을 어느 인코딩을 시도해도 깨져서 나옴
(파일 경로 : 카카오톡 다운로드, google drive)

## 해결책
csv 파일 메모장으로 열기
-> 다른 이름으로 저장, 인코딩 utf-8 명시하여 덮어쓰기
-> pandas에서 encoding='utf-8-sig'으로 읽기