# Daily Journal 운영 가이드라인

## 목적
- Hugging Face Daily Papers 결과를 매일 일관된 형식으로 기록하고,
- Discord 브리핑과 분리해서 `aistudy` 저장소에 축적하며,
- 자동 커밋/푸시까지 안정적으로 수행한다.

## 기본 경로 규칙
- 출력 루트: `papers/daily-journal`
- 날짜 경로: `papers/daily-journal/YYYY/MM/YYYY-MM-DD.md`
- 예시: `papers/daily-journal/2026/02/2026-02-15.md`

## 문서 형식 규칙
1. 제목
   - `# Hugging Face Daily Papers Journal — YYYY-MM-DD`
2. 메타
   - `Source mode`
   - `Total papers`
3. 본문
   - 추천 순위 기준 상위 N개(기본 5개)
   - 각 항목: 제목, 점수, 링크, 핵심 요약

## 품질 규칙
- 요약은 단순 번역보다 "핵심 아이디어 + 의미" 중심으로 작성.
- 점수/링크 누락 금지.
- 생성 실패 시 빈 파일 생성 금지.
- fallback(mock/API 장애) 발생 시 메타에 mode를 명시.

## 변경/보존 규칙
- 기존 날짜 파일 전체 덮어쓰기 지양.
- 자동화는 동일 날짜 재실행 시 내용 일관성을 유지.
- 생성물 외 불필요 파일(임시파일/로그) 커밋 금지.

## 자동화 운영 규칙
- 매일 KST 기준 1회 실행.
- 파이프라인 실행 후 `papers/daily-journal` 변경분만 커밋.
- 변경 있을 때만 푸시.
- 커밋 메시지는 건조하고 명확하게 유지.
  - 예: `chore(daily-journal): publish 2026-02-15`

## 장애 대응
- API 실패/권한 오류 시 에러 요약만 알림 채널에 보고.
- 다음 스케줄 주기에서 자동 재시도(수동 개입 최소화).

## 메모
- 브리핑/회고 자동화와 journal 자동화는 분리 운영 가능.
- journal은 저장소 기록(artifact), 브리핑은 커뮤니케이션(요약) 역할로 분리한다.
