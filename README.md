# 맞춤형 LLM 개발 학습 커리큘럼 — 24주

> Apple Silicon Mac(M2 / 32GB)과 Raspberry Pi 4(4GB)만으로 LLM의 원리부터 모델 구축·훈련, Fine-tuning, 평가, 로컬 서빙, 엣지 배포까지 실습하는 24주 학습 계획.

현재 환경이라면 추가 하드웨어 구매 없이 전체 LLM 개발 파이프라인을 실습할 수 있슴. 다만 대규모 모델의 사전학습이나 수십억 파라미터 모델의 전체 파라미터 학습은 현재 장비만으로는 현실적이지 않으므로, 학습 목표에 맞게 모델 규모를 구분한다.

## 목차

1. [개발환경과 학습 방향](#1-개발환경과-학습-방향)
2. [24주 상세 학습 일정](#2-24주-상세-학습-일정)
3. [추천 도서와 읽기 일정](#3-추천-도서와-읽기-일정)
4. [첫 주 실습 — Mac M2 개발환경 구성](#4-첫-주-실습--mac-m2-개발환경-구성)
5. [단계별 성취 기준](#5-단계별-성취-기준)
6. [최종 프로젝트 목표](#6-최종-프로젝트-목표)
7. [추가로 추천하는 학습 방식](#7-추가로-추천하는-학습-방식)
8. [최종 권장 사항](#8-최종-권장-사항)

---

## 1. 개발환경과 학습 방향

| 항목 | 현재 환경 | 활용 계획 |
|---|---|---|
| 개발 PC | Apple Silicon M2 / 32GB | 모델 구현, 훈련, Fine-tuning, 성능 분석 |
| 엣지 디바이스 | Raspberry Pi 4 / 4GB | 양자화 모델 실행, ARM 최적화, 배포 |
| 딥러닝 수준 | 신경망·머신러닝 기초 이해 | 기초 수학 복습 최소화, PyTorch 집중 |
| 프로그래밍 | Python, C/C++ | 모델 개발과 추론 엔진 통합 |
| 최종 목표 | 모델 개발부터 서비스 구축까지 | 전체 LLM 개발 파이프라인 구현 |

### 핵심 개발 도구

Mac에서는 다음 세 가지를 병행.

- **PyTorch + MPS**: Transformer 직접 구현 및 소형 모델 훈련
- **MLX / MLX-LM**: Apple Silicon에서 기존 LLM 실행 및 LoRA Fine-tuning
- **llama.cpp**: C/C++ 추론 엔진 분석, 양자화, 로컬 및 Raspberry Pi 배포

> MLX는 Apple Silicon의 통합 메모리 구조를 활용하도록 설계된 프레임워크이며, MLX-LM은 언어모델 생성과 Fine-tuning을 지원함.

### 학습 프로젝트

| 프로젝트 | 실행 환경 | 결과물 |
|---|---|---|
| **A. Mini GPT** | Mac M2 | 직접 구축·훈련한 소형 언어모델 |
| **B. Domain LLM** | Mac M2 | Fine-tuning된 산업용 질의응답 모델 |
| **C. Edge LLM** | Mac + Raspberry Pi | 로컬 AI 진단 서비스 |

---

## 2. 24주 상세 학습 일정

주당 **10~12시간**, 총 **240~288시간** 기준.

### 1단계 — PyTorch와 Transformer 기초 (1~4주)

이미 신경망과 머신러닝의 기본개념을 이해하고 있다는 전제 하에 PyTorch의 모델 구현과 학습 과정에 집중.

| 주차 | 이론 학습 | 실습 과제 | 완료 기준 |
|---|---|---|---|
| 1주 | Tensor, Autograd, Backpropagation | PyTorch 신경망 구현 | 학습·검증 Loss 시각화 |
| 2주 | Tokenization, BPE, Embedding | 간단한 Tokenizer 구현 | 텍스트 인코딩·디코딩 |
| 3주 | Self-Attention, Q/K/V, Causal Mask | Attention 직접 구현 | Attention 출력 검증 |
| 4주 | Multi-Head Attention, Transformer Block | Decoder Block 구현 | 입력·출력 차원 및 마스킹 테스트 |

- **주교재**: 『밑바닥부터 만들면서 배우는 LLM』 1~4장
- **공식 예제**: [LLMs from Scratch](https://github.com/rasbt/LLMs-from-scratch) — 장별 실습 코드와 Attention, GPT 구현 예제 제공

**4주차 완료 결과물**

```
mini-gpt/
├── tokenizer.py
├── embedding.py
├── attention.py
├── transformer.py
└── tests/
    ├── test_tokenizer.py
    ├── test_attention.py
    └── test_transformer.py
```

> 💡 단순히 코드가 실행되는 것보다 Attention 계산 결과와 Causal Mask가 정확한지 **단위 테스트로 검증**하는 것이 중요.

### 2단계 — LLM 구축과 사전학습 (5~8주)

이 단계에서는 모델을 처음부터 구성하고 훈련한다.

| 주차 | 학습 내용 | 실습 | 결과물 |
|---|---|---|---|
| 5주 | GPT 아키텍처, 모델 파라미터 | Mini GPT 완성 | Decoder-only 모델 |
| 6주 | Dataset, DataLoader, AdamW | 학습 파이프라인 | Training Loop |
| 7주 | Pre-training, Checkpoint | 텍스트 데이터 사전학습 | 모델 가중치 |
| 8주 | Perplexity, Validation, Sampling | 모델 평가 및 텍스트 생성 | Mini GPT v1 |

**권장 모델 규모 (Mac M2 32GB)**

| 항목 | 권장 설정 |
|---|---|
| Transformer Layers | 4 |
| Embedding Dimension | 256 |
| Attention Heads | 4 |
| Context Length | 256 |
| Vocabulary | 8,000 |
| Batch Size | 4~16 |
| Framework | PyTorch |
| Device | MPS |

이 구성은 대략 수백만 파라미터 규모이며, 정확한 수는 구현 방식에 따라 달라짐.

처음에는 아주 작은 데이터셋으로 과적합이 가능한지 확인하고, 이후 데이터 규모를 늘려 학습한다. 학습 목표는 뛰어난 언어 생성 성능이 아니라 **전체 훈련 파이프라인의 정확한 이해**임.

**평가 지표**

- Training Loss
- Validation Loss
- Perplexity
- Training Time
- Peak Memory
- Tokens/sec

- **주교재**: 『밑바닥부터 만들면서 배우는 LLM』 5장
- **심화 강의**: [Stanford CS336 — Language Modeling from Scratch](https://stanford-cs336.github.io/)

> Stanford CS336에는 Tokenizer, Transformer, Optimizer 구현부터 데이터 처리, 평가, 최적화까지 포함되어 있습니다. 다만 일부 과제는 NVIDIA GPU 및 대규모 연산 환경을 전제로 하므로 Mac에서는 기본 구현 과제를 중심으로 진행하는 것이 좋음.

### 3단계 — 기존 LLM 활용 및 Fine-tuning (9~12주)

여기서부터 실무에 사용할 수 있는 모델을 다룬다.

| 주차 | 학습 내용 | 실습 | 결과물 |
|---|---|---|---|
| 9주 | Hugging Face, Model Hub, Safetensors | 0.5B~1.5B 모델 실행 | Base Model |
| 10주 | SFT, LoRA, QLoRA | LoRA 학습 | Adapter |
| 11주 | Dataset 구축, Instruction Tuning | 산업용 데이터 Fine-tuning | Domain Model |
| 12주 | 모델 평가, Hallucination | Base/Fine-tuned 비교 | 평가 보고서 |

**Mac M2에 적합한 실습 전략**

1. 0.5B~1.5B 모델로 전체 Fine-tuning 절차를 익힌다.
2. MLX-LM으로 LoRA 학습을 수행.
3. 여유 메모리와 학습 시간을 확인한 후 3B 모델까지 확장.
4. 모델별 성능을 비교.

**산업용 데이터셋 예제**

다음 형식의 데이터를 구성한다.

```json
{
  "messages": [
    {
      "role": "user",
      "content": "압력 센서 값이 정상 범위보다 낮을 때 점검할 항목은?"
    },
    {
      "role": "assistant",
      "content": "압력 공급 상태, 센서 연결, 밸브 상태 및 설정값을 점검해야 합니다."
    }
  ]
}
```

- 실습용으로는 **300~1,000개의 검증된 질의응답 데이터**부터 시작. 이 수량은 학습 파이프라인 검증을 위한 기준이지 충분한 모델 성능을 보장하는 규모는 아님.
- 학습 데이터와 평가 데이터는 **반드시 분리**해야 함.

**추천 자료**: [Hugging Face LLM Course](https://huggingface.co/learn/llm-course)

### 4단계 — 양자화와 로컬 추론 (13~16주)

이 단계부터 C/C++ 기반 추론 엔진을 본격적으로 다룬다.

| 주차 | 학습 내용 | 실습 | 결과물 |
|---|---|---|---|
| 13주 | FP16, INT8, INT4, GGUF | 모델 양자화 | Q4/Q8 모델 |
| 14주 | llama.cpp 구조 | 소스 빌드 및 실행 | C++ 추론 환경 |
| 15주 | KV Cache, Prefill, Decode | 추론 성능 측정 | Benchmark |
| 16주 | C API, Memory Management | C++ 애플리케이션 연동 | Local LLM Client |

**llama.cpp 학습 순서**

```
GGUF Model
    │
    ▼
Model Loader
    │
    ▼
Tokenizer
    │
    ▼
Inference Context
    │
    ├── Transformer Layers
    ├── KV Cache
    └── Backend (CPU / Metal)
    │
    ▼
Sampler
    │
    ▼
Generated Tokens
```

단순히 llama.cpp를 실행하는 것에 그치지 않고, 내부에서 모델을 로딩하고 토큰을 생성하는 과정을 분석한다.

**공식 저장소**: [llama.cpp](https://github.com/ggml-org/llama.cpp)

**성능 비교** — 동일한 모델을 FP16, Q8, Q4로 실행하고 다음을 측정.

| 지표 | 측정 목적 |
|---|---|
| TTFT | 첫 토큰 응답 지연 |
| Decode Tokens/sec | 생성 속도 |
| Peak RSS | 최대 메모리 사용량 |
| Model Load Time | 초기화 시간 |
| CPU/GPU Utilization | 하드웨어 사용률 |
| Task Accuracy | 양자화에 따른 품질 변화 |

> 가능하면 동일한 프롬프트, Context Length, 출력 토큰 수를 사용하여 비교한다.

### 5단계 — LLM 서빙과 RAG (17~20주)

이 단계에서는 모델을 실제 서비스 형태로 구성한다.

| 주차 | 학습 내용 | 실습 | 결과물 |
|---|---|---|---|
| 17주 | REST API, Streaming | llama.cpp 서버 실행 | 추론 API |
| 18주 | FastAPI, 동시 요청 처리 | API Gateway 구현 | LLM 서비스 |
| 19주 | Embedding, Vector Search, RAG | 장비 매뉴얼 검색 | RAG 서비스 |
| 20주 | Docker, Logging, Monitoring | 통합 서버 구축 | 운영 가능한 서비스 |

**권장 아키텍처**

```
┌─────────────────────────┐
│      Web / C++ GUI      │
└────────────┬────────────┘
             │ REST API
             ▼
┌─────────────────────────┐
│        FastAPI          │
│                         │
│ Authentication          │
│ Request Validation      │
│ Streaming               │
└────────────┬────────────┘
             │
       ┌─────┴─────┐
       ▼           ▼
┌────────────┐ ┌──────────────┐
│ RAG Engine │ │ LLM Server   │
│            │ │              │
│ Embedding  │ │ llama.cpp    │
│ Vector DB  │ │ MLX-LM       │
└────────────┘ └──────────────┘
```

- Mac M2에서는 llama.cpp 서버와 MLX-LM 서버를 각각 실습.
- 서빙 성능은 요청 수, 프롬프트 길이, 생성 토큰 수, 동시 사용자 수에 따라 측정.

### 6단계 — Raspberry Pi 엣지 배포 (21~24주)

최종 단계에서는 Raspberry Pi 4 4GB에 모델을 배포한다.

| 주차 | 학습 내용 | 실습 | 결과물 |
|---|---|---|---|
| 21주 | ARM 아키텍처, Linux 최적화 | llama.cpp ARM 빌드 | Pi 추론 환경 |
| 22주 | 저메모리 추론, Quantization | 0.5B~1B 모델 실행 | Edge LLM |
| 23주 | API 서버, 서비스 자동 실행 | systemd 서비스 구성 | Edge API |
| 24주 | 통합 테스트, 성능 평가 | 산업용 AI Assistant | 최종 프로젝트 |

**Raspberry Pi 4 권장 모델**

| 모델 규모 | Q4 가중치 용량(대략) | 권장 여부 |
|---|---|---|
| 0.5B | 0.3~0.5GB | 우선 실습 |
| 1B | 0.6~0.9GB | 권장 |
| 1.5B | 0.9~1.3GB | 성능 측정용 |
| 3B | 1.8~2.5GB | 제한적 실험 |

> ⚠️ 위 수치는 모델 구조와 양자화 형식에 따라 달라지는 대략적인 가중치 크기임. 실행 시에는 KV Cache 및 기타 버퍼가 추가로 필요함.

Raspberry Pi 4에서는 CPU 추론을 사용하므로 실시간 대화에 충분한 응답 속도를 얻지 못할 수도 있슴. 따라서 0.5B~1B 모델부터 시작하고, 응답 속도와 메모리 사용량을 기준으로 모델을 선택한다.

---

## 3. 추천 도서와 읽기 일정

책을 많이 구입하기보다 두 권을 중심으로 학습하는 방식으로 진행.

| 도서 / 자료 | 활용 시기 | 우선순위 |
|---|---|---|
| 『밑바닥부터 만들면서 배우는 LLM』 | 1~12주 | **필수** |
| 『트랜스포머를 활용한 자연어 처리』 | 9~16주 | 권장 |
| 『밑바닥부터 시작하는 딥러닝 2』 | 필요할 때 | 보충 |
| Stanford CS336 강의자료 | 5~12주 | 심화 |
| llama.cpp 공식 문서 | 13~24주 | **필수** |

특히 Sebastian Raschka의 『밑바닥부터 만들면서 배우는 LLM』을 주교재로 삼는다. [공식 저장소](https://github.com/rasbt/LLMs-from-scratch)에는 2장 Tokenization, 3장 Attention, 4장 GPT 구현, 5장 사전학습, 6~7장 Fine-tuning 코드가 정리되어 있다.

---

## 4. 첫 주 실습 — Mac M2 개발환경 구성

첫 주에는 PyTorch와 MLX를 사용할 수 있는 환경을 구축.

### 4.1 Python 가상환경

```bash
mkdir -p ~/projects/llm-study
cd ~/projects/llm-study
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 4.2 PyTorch 설치

```bash
pip install torch torchvision
```

설치 후 MPS 지원을 확인합니다.

```python
import torch
print("PyTorch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
```

`MPS available: True`가 출력되면 Apple GPU를 사용할 수 있다.

### 4.3 MLX 설치

```bash
pip install mlx mlx-lm
```

MLX는 Apple Silicon에서 모델 실행과 Fine-tuning을 위한 주요 실습 도구.

**공식 문서**: [MLX Framework](https://ml-explore.github.io/mlx/)

### 4.4 학습 프로젝트 디렉터리

전체 24주 동안 다음 구조를 사용한다.

```
llm-study/
│
├── 01-pytorch/
│   ├── tensor/
│   ├── autograd/
│   └── neural-network/
│
├── 02-transformer/
│   ├── tokenizer/
│   ├── attention/
│   └── transformer-block/
│
├── 03-mini-gpt/
│   ├── model/
│   ├── training/
│   ├── evaluation/
│   └── checkpoints/
│
├── 04-fine-tuning/
│   ├── datasets/
│   ├── lora/
│   └── evaluation/
│
├── 05-inference/
│   ├── mlx/
│   ├── llama-cpp/
│   └── benchmarks/
│
├── 06-serving/
│   ├── api/
│   ├── rag/
│   └── monitoring/
│
└── 07-edge/
    ├── raspberry-pi/
    ├── deployment/
    └── benchmarks/
```

---

## 5. 단계별 성취 기준

다음 기준을 만족하면 다음 단계로 넘어간다.

| 단계 | 완료 기준 |
|---|---|
| PyTorch | 신경망 훈련 코드를 독립적으로 작성 |
| Transformer | Self-Attention과 Decoder Block 직접 구현 |
| Mini GPT | 모델 사전학습 및 텍스트 생성 성공 |
| Fine-tuning | LoRA 적용 전후 성능 비교 |
| Quantization | Q4/Q8 모델 생성 및 품질 비교 |
| llama.cpp | C++ 애플리케이션에서 모델 추론 |
| Serving | Streaming API와 동시 요청 처리 |
| RAG | 장비 문서 기반 질의응답 구현 |
| Edge | Raspberry Pi에서 모델 실행 |
| 최종 프로젝트 | Mac과 Raspberry Pi 성능 비교 및 통합 서비스 완성 |

**진행 체크리스트**

- [ ] PyTorch
- [ ] Transformer
- [ ] Mini GPT
- [ ] Fine-tuning
- [ ] Quantization
- [ ] llama.cpp
- [ ] Serving
- [ ] RAG
- [ ] Edge
- [ ] 최종 프로젝트

---

## 6. 최종 프로젝트 목표

### Industrial Edge LLM Assistant

```
Industrial Equipment
        │
        │ MQTT / Ethernet
        ▼
┌──────────────────────┐
│ Data Collector       │
│ Python / C++         │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ State Analyzer       │
│ Alarm Processor      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Edge LLM             │
│ Raspberry Pi 4       │
│ llama.cpp            │
│ Quantized Model      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ REST API             │
│ FastAPI              │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Monitoring GUI       │
│ React / C++ Qt       │
└──────────────────────┘
```

- LLM은 장비 상태와 알람을 자연어로 설명하고, 정비 문서를 검색하여 점검 방법을 안내한다.
- 모델 추론 성능이 제한적인 Raspberry Pi에서는 데이터 수집과 상태 분석을 별도 프로세스로 유지하고, LLM 추론은 **비동기**로 처리한다.
- 최종적으로 Mac M2와 Raspberry Pi 4에서 동일한 모델을 실행해 성능을 비교한다.

---

## 7. 추가로 추천하는 학습 방식

두 가지 학습 트랙을 병행하는 것이 가장 효과적.

| 기간 | 모델 개발 트랙 | 시스템 개발 트랙 |
|---|---|---|
| 1~4주 | PyTorch, Transformer | Mac에서 로컬 LLM 실행 |
| 5~8주 | Mini GPT 훈련 | 추론 성능 측정 |
| 9~12주 | Fine-tuning | MLX-LM 활용 |
| 13~16주 | 모델 양자화 | llama.cpp, C++ |
| 17~20주 | RAG 및 평가 | FastAPI, 서빙 |
| 21~24주 | 엣지 모델 최적화 | Raspberry Pi 배포 |

모델 학습 이론과 실제 추론 시스템 개발을 연결하면서 진행할 수 있슴.

---

## 8. 최종 권장 사항

- **Mac M2 32GB**를 주 개발·훈련 장비로, **Raspberry Pi 4 4GB**를 최종 배포 및 최적화 장비로 사용.
- 첫 8주는 Transformer와 소형 언어모델 구현에 집중하고, 이후 기존 모델의 Fine-tuning과 최적화에 시간을 더 많이 배분.
- C/C++ 기반 추론 엔진인 llama.cpp의 구조와 메모리 관리, KV Cache, ARM 최적화까지 학습하면 기존 산업용 소프트웨어 개발 역량과도 연결할 수 있슴.

---

## 참고 자료

- [LLMs from Scratch (Sebastian Raschka)](https://github.com/rasbt/LLMs-from-scratch)
- [Stanford CS336 — Language Modeling from Scratch](https://stanford-cs336.github.io/)
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [MLX Framework](https://ml-explore.github.io/mlx/)
- [MLX-LM](https://github.com/ml-explore/mlx-lm)
