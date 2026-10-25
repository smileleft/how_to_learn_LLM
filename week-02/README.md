# 02주차 — 문자 토큰화와 임베딩

[전체 목차](README.md) · [이전 주](week-01.md) · [다음 주](week-03.md)

## 이번 주 목표

**문자 토큰화와 nn.Embedding**. 학습 결과를 `~/projects/llm-study/week-02/`에 저장한다.

## 일자별 학습 일정 (약 10~12시간)

- **월: 핵심 개념 및 용어 정리 (1시간)**
- **화: 교재와 공식 문서 읽기 (1~1.5시간)**
- **수: 코드 직접 입력 및 실행 (1~1.5시간)**
- **목: 단위 테스트·변형 실험 (1~1.5시간)**
- **금: 결과 기록 및 질문 풀이 (1시간)**
- **토: 주간 통합 과제 구현 (3~4시간)**
- **일: 코드 정리·회고·다음 주 준비 (1~1.5시간)**

## 교재 읽을 범위

주교재 2장: 텍스트 데이터 처리 및 토큰화; Hugging Face Tokenizers 개요


## 실습 준비

```bash
cd ~/projects/llm-study
source .venv/bin/activate
mkdir -p week-02
cd week-02
```

아래 코드를 `main.py`로 저장하고 `python main.py`로 실행.

## 실습 코드 — 최소 재현 실험

```python
import torch
from torch import nn
text='hello edge llm'; vocab={c:i for i,c in enumerate(sorted(set(text)))}
ids=torch.tensor([vocab[c] for c in text]); emb=nn.Embedding(len(vocab),8)
print('vocab:',len(vocab));print('embedding shape:',tuple(emb(ids).shape))
```

## 예상 실행 결과

```text
vocab: 9 / embedding shape: (14, 8)
```


## 주말 통합 과제

1. 최소 실험을 독립적으로 다시 작성하고 각 단계에 주석을 추가한다.
2. 입력 크기 또는 설정을 2가지 이상 바꾸어 결과 차이를 표로 기록한다.
3. 해당 주제의 기능을 이전 주 프로젝트와 연결합니다. 통합이 불가능한 경우 함수/API 계약과 테스트를 작성한다.
4. 실행 명령, 소요 시간, 오류 및 해결 방법을 `NOTES.md`에 기록한다.
5. `pytest` 기반 테스트를 최소 2개 작성(정상 동작 및 경계 조건).

## 확인 문제

1. 문자 단위와 BPE 토큰화 차이는?
2. 어휘 크기가 커지면 임베딩 메모리는?
3. 토큰 ID가 의미적 거리를 나타내는가?

## 완료 체크리스트

- [ ] 교재와 문서의 지정 주제 학습
- [ ] 예제 실행 및 예상 결과 확인
- [ ] 입력/파라미터 변경 실험 2건
- [ ] 테스트 2개 이상 통과
- [ ] 확인 문제 답안 작성
- [ ] `NOTES.md`에 실험 결과와 한계 기록

## 다음 주 연결

3주차에는 Self-Attention 직접 구현을(를) 학습한다.
