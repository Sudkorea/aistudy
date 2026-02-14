# RAG 적용 포트폴리오: Repos 지식 연결 + Agent Memory Recall 강화

## 1. 목표
이번 작업의 목적은 두 가지였다.

1) `aistudy` + `obsidian-vault` 지식을 안전하게 연결하는 RAG 파이프라인 구축  
2) `memory_search` 인증 이슈가 있어도 에이전트가 과거 맥락을 안정적으로 회수하도록 로컬 메모리 검색 fallback 구축

핵심은 **정확도(기억/검색)와 개인정보 안전성(PII 최소 노출)**을 동시에 확보하는 것이다.

---

## 2. Repos RAG 적용 (지식 연결 파이프라인)

### 2.1 문제 정의
- `aistudy`는 학습/기술 지식이 풍부하지만 개인 맥락이 약함
- `obsidian-vault`는 개인 경험/스토리가 풍부하지만 민감정보가 섞여 있음
- 둘을 그대로 합치면 검색 품질은 올라가도 개인정보 노출 반경이 커짐

### 2.2 적용 전략: 2+1
- 원본 private vault (`obsidian-vault/main`)는 보존
- RAG 전용 안전 브랜치/워크트리(`obsidian-vault-rag-safe`) 분리
- sanitize 스크립트로 PII 마스킹 후, RAG는 `aistudy + rag-safe`만 사용

### 2.3 구현 포인트
- `sanitize_obsidian_for_rag.py`
  - 이메일/전화/연락처/주소/토큰 패턴 치환
  - `.obsidian`, airlock, 개인성 높은 일부 경로 제외
- `rag_sources.json`
  - 허용 소스: `aistudy`, `obsidian-vault-rag-safe`
  - 차단 소스: raw `obsidian-vault`
- 정기 sanitize cron
  - 주기 갱신 + 변경 시 커밋(푸시는 수동)

### 2.4 검증 결과
- raw index의 연락처가 rag-safe에서 `[REDACTED_CONTACT]`로 치환됨
- 검색 결과에 raw 경로 대신 rag-safe 경로가 노출됨
- RAG 품질은 유지하면서 민감정보 노출면을 낮춤

---

## 3. Agent Memory Recall용 RAG 적용 (local fallback)

### 3.1 문제 정의
- OpenClaw `memory_search`가 provider auth(openai/google/voyage) 미설정 상태에서 실패
- 결과적으로 "이전 작업/결정 회수"가 불안정해질 수 있음

### 3.2 해결 전략
- 외부 API 0 의존 로컬 recall 체계 추가
- 1순위: `memory_search` 시도
- 실패 시: 로컬 메모리 검색기로 자동 fallback

### 3.3 구현 포인트
- `local_memory_search.py`
  - 대상: `MEMORY.md`, `memory/*.md`
  - sparse 환경 fallback: `AGENTS.md`, `SOUL.md`, `USER.md`, `CLAUDE.md` 등
  - 출력: `path#line + snippet + score`
- `local_memory_get.py`
  - 파일 스니펫 라인 단위 조회
- `recall_memory.sh`
  - `openclaw memory search` 실패/빈결과 시 local search 자동 전환
- `AGENTS.md` 규칙 업데이트
  - prior-work 질문 응답 시 recall 프로토콜 강제

### 3.4 운영 방식
- 사용자 응답에는 기본적으로 출처를 노출하지 않음
- 근거 확인을 요청받을 때만 `path#line` 공개
- 목적은 "사용자 검열"이 아니라 "에이전트 헛기억 방지"

---

## 4. 시행착오와 개선 포인트

### 시행착오
- 초기 자동화는 기능 달성에 집중해 경계조건(권한/범위 제한)에서 실수가 발생
- memory 검색은 모델 경로와 메모리 도구 경로 인증이 분리되어 있다는 점을 늦게 확인

### 개선
- Codex-safe 모드 도입(계획→최소패치→검증→롤백)
- RAG 소스 분리 정책 명시
- 메모리 recall fallback으로 장애 내성 확보

---

## 5. 기술적 의미
- 단순 챗봇이 아니라, **운영 가능한 지식 시스템**으로 확장
- 개인정보 보호와 지식 활용을 트레이드오프가 아닌 구조적 분리로 해결
- 외부 인증 장애가 있어도 local recall로 기능 연속성 유지

---

## 6. 한 줄 요약
**Repos RAG는 안전하게 넓히고, Memory RAG는 로컬 fallback으로 끊김 없이 유지했다.**  
결과적으로 "지식 연결성 + 운영 안정성 + 개인정보 보호"를 동시에 달성했다.
