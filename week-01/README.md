# 1주차 상세 학습계획 — PyTorch 기초와 개발환경

> 작성일: 2026-10-27

## 1주차 개요

1주차의 목표는 Mac M2에 PyTorch(MPS)·MLX 환경을 갖추고, Tensor → Autograd → 역전파 → 신경망 학습까지 직접 구현해 **학습·검증 Loss 곡선을 그리는 것**이다. 저장소 README의 1주차 정의(이론: Tensor, Autograd, Backpropagation / 실습: PyTorch 신경망 구현 / 완료 기준: 학습·검증 Loss 시각화)와 4장 '첫 주 실습'을 하나로 묶었어서 진행한다.

- **전제**: 신경망·머신러닝 기초 개념은 알고 있음, Python 사용 가능
- **시간**: 주 10~12시간
- **병행 트랙**: 시스템 개발 트랙(Mac에서 로컬 LLM 실행)을 1주차에 가볍게 시작해, 앞으로 만들 모델이 실제로 어떻게 돌아가는지 먼저 체감한다

| 영역 | 배분 시간 | 비중 |
| --- | --- | --- |
| 환경 구성 | 1.5시간 | 13% |
| 이론 (Tensor·Autograd·역전파) | 3시간 | 26% |
| 실습 (구현·학습·시각화) | 5시간 | 44% |
| 시스템 트랙 (MLX-LM 실행) | 1시간 | 9% |
| 정리·회고 | 1시간 | 9% |
| **합계** | **11.5시간** | |

## 사전 준비와 개발환경 구성

환경 구성은 Day 1에 끝내고, 아래 체크리스트가 모두 통과해야 이론·실습으로 넘어간다. Apple Silicon에서 자주 생기는 문제(Rosetta로 실행된 x86 Python)를 막는 확인 단계를 추가함.

**1) 가상환경과 패키지 설치** — Python 3.11 또는 3.12 권장

```bash
mkdir -p ~/projects/llm-study && cd ~/projects/llm-study
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision mlx mlx-lm
pip install numpy matplotlib scikit-learn jupyter tqdm
pip freeze > requirements.txt
```

**2) 환경 검증 스크립트** — `01-pytorch/check_env.py`로 저장

```python
import platform, time, torch
import mlx.core as mx

print("Machine:", platform.machine())          # arm64 여야 함 (x86_64면 Rosetta Python)
print("Python :", platform.python_version())
print("PyTorch:", torch.__version__, "| MPS:", torch.backends.mps.is_available())
print("MLX    :", mx.default_device())

# CPU vs MPS 행렬곱 속도 비교
for dev in ["cpu", "mps"]:
    x = torch.randn(4096, 4096, device=dev)
    torch.mps.synchronize() if dev == "mps" else None
    t = time.time()
    for _ in range(10):
        y = x @ x
    torch.mps.synchronize() if dev == "mps" else None
    print(f"{dev}: {time.time() - t:.2f}s")
```

**3) 체크리스트**

- [ ] `platform.machine()` 출력이 `arm64`
- [ ] `MPS: True` 출력
- [ ] MLX 기본 디바이스가 `Device(gpu, 0)`
- [ ] CPU 대비 MPS 행렬곱이 더 빠른 것을 숫자로 확인하고 기록
- [ ] `requirements.txt` 생성, Git 저장소 초기화 (`.venv/`는 `.gitignore`에 추가)
- [ ] Jupyter 또는 VS Code에서 `.venv` 커널 선택 확인

> 문제가 생기면: `MPS: False`이거나 machine이 `x86_64`라면 Homebrew 또는 python.org의 arm64 Python으로 venv를 다시 만든다.

## 일별 학습 계획 (Day 1~7)

흐름은 **환경 → Tensor → 미분·경사하강 → 역전파 직접 구현 → PyTorch Autograd → 신경망 학습 → Loss 시각화** 순서다. 역전파를 먼저 손으로 구현한 뒤 Autograd로 넘어가야 4주차 Attention 구현 때 gradient 문제를 스스로 디버깅할 수 있다.

| Day | 주제 | 이론 (읽기) | 실습 | 산출물 | 시간 |
| --- | --- | --- | --- | --- | --- |
| 1 | 환경 구성 + Tensor 입문 | LLMs-from-scratch 부록 A 앞부분 (PyTorch 소개, Tensor) | 환경 검증 스크립트 실행, Tensor 생성·shape·dtype·device 실습 | `check_env.py`, 환경 기록 | 2h |
| 2 | Tensor 연산과 경사하강 | 밑바닥 딥러닝 4장 (수치미분, 기울기, 경사하강) | broadcasting·reshape·matmul 연습, NumPy로 선형회귀 경사하강 구현 | `tensor_basics.ipynb`, `numpy_linreg.py` | 1.5h |
| 3 | 계산 그래프와 역전파 | 밑바닥 딥러닝 5장 (연쇄법칙, 덧셈·곱셈·활성화 노드) | 스칼라 자동미분 엔진(`Value` 클래스) 직접 구현, 수치미분과 비교 | `mini_autograd.py` | 1.5h |
| 4 | PyTorch Autograd | 부록 A의 Autograd 절, PyTorch Autograd 튜토리얼 | Day 3 결과를 `torch.autograd`와 비교, grad 누적·`zero_grad`·`no_grad`·`detach` 실험 | `autograd_check.py` | 1.5h |
| 5 | 신경망 구현과 학습 루프 | 부록 A의 nn.Module·DataLoader·학습 루프 절 | make_moons 데이터로 MLP 분류기, train/val 분리, MPS에서 학습 | `mlp_train.py` | 2h |
| 6 | Loss 시각화와 실험 | — | Loss 곡선 그리기, 학습률·모델 크기·과적합 실험 3종 | `loss_curve.png`, 실험 표 | 1.5h |
| 7 | 시스템 트랙 + 회고 | MLX-LM README | MLX-LM으로 소형 LLM 실행·속도 측정, 1주차 회고 작성 | `mlx_first_run.md`, `week01_retro.md` | 1.5h |

**하루 진행 방식 (공통)**

1. 읽기 30~40분: 교재 범위를 읽고 핵심 개념을 3줄로 요약
2. 구현 60분: 교재 코드를 복사하지 않고 직접 타이핑, 막히면 그때 확인
3. 기록 10분: `NOTES.md`에 '오늘 배운 것 / 막힌 것 / 내일 확인할 것' 작성 후 Git 커밋

**Day 2 Tensor 연습 목록** — 각 연산의 결과 shape을 실행 전에 먼저 예측해 본다.

- `torch.zeros`, `ones`, `randn`, `arange`와 `dtype`(float32/float16/int64) 변환
- `view` vs `reshape`, `transpose`·`permute`, `unsqueeze`·`squeeze`
- broadcasting: `(B, T, C) + (C,)`, `(B, T, 1) * (B, T, C)` — 3~4주차 Attention에서 그대로 쓰이는 형태
- `@`, `torch.matmul`, `torch.bmm`로 배치 행렬곱
- `.to("mps")`로 디바이스 이동, CPU 텐서와 섞었을 때 나는 에러 직접 확인

## 핵심 실습 과제 상세

세 개의 코드가 1주차의 뼈대: 역전파를 직접 구현하는 `mini_autograd.py`, PyTorch와 비교하는 `autograd_check.py`, Loss를 시각화하는 `mlp_train.py`. 아래 코드는 출발점이며 TODO는 직접 채운다.

### 과제 1 — 스칼라 자동미분 엔진 (Day 3)

목표: 계산 그래프를 만들고 위상 정렬 역순으로 기울기를 전파하는 과정을 30~60줄로 직접 구현한다.

```python
# 01-pytorch/autograd/mini_autograd.py
import math

class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data, self.grad = data, 0.0
        self._backward = lambda: None
        self._prev, self._op = set(_children), _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad          # 덧셈 노드: 기울기를 그대로 전달
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad   # 곱셈 노드: 상대 입력을 곱해 전달
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    # TODO: __radd__, __rmul__, __neg__, __sub__, __pow__, relu, exp

    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._prev:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

# 검증: y = tanh(x*w + b)의 dy/dw를 수치미분과 비교
x, w, b = Value(2.0), Value(-3.0), Value(1.0)
y = (x * w + b).tanh()
y.backward()
f = lambda w_: math.tanh(2.0 * w_ + 1.0)
h = 1e-6
print("autograd:", w.grad, "| numeric:", (f(-3.0 + h) - f(-3.0 - h)) / (2 * h))
```

확장 과제: 같은 변수를 두 번 쓰는 식(`a = x * x`)에서 `+=`가 아닌 `=`로 기울기를 저장하면 왜 틀리는지 확인한다.

### 과제 2 — PyTorch Autograd와 비교 (Day 4)

```python
# 01-pytorch/autograd/autograd_check.py
import torch
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(-3.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
y = torch.tanh(x * w + b)
y.backward()
print("torch:", w.grad.item())   # 과제 1의 w.grad와 일치해야 함

# 실험: backward를 두 번 하면? (retain_graph 필요성과 grad 누적 확인)
# 실험: torch.no_grad() 안에서 계산하면 grad_fn이 어떻게 되는가?
# 실험: y.detach()와 y.data의 차이
```

### 과제 3 — MLP 학습과 Loss 시각화 (Day 5~6)

2차원 make_moons 데이터는 1초 안에 만들어지고 결정 경계를 눈으로 볼 수 있어 첫 학습 파이프라인 검증에 적합.

```python
# 01-pytorch/neural-network/mlp_train.py
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

torch.manual_seed(42)
device = "mps" if torch.backends.mps.is_available() else "cpu"

X, y = make_moons(n_samples=2000, noise=0.2, random_state=42)
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)
ds = lambda X_, y_: TensorDataset(torch.tensor(X_, dtype=torch.float32),
                                  torch.tensor(y_, dtype=torch.long))
train_dl = DataLoader(ds(X_tr, y_tr), batch_size=64, shuffle=True)
val_dl = DataLoader(ds(X_va, y_va), batch_size=256)

class MLP(nn.Module):
    def __init__(self, d_in=2, d_hidden=64, d_out=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )
    def forward(self, x):
        return self.net(x)

model = MLP().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

def run_epoch(dl, train):
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total += loss.item() * len(xb)
            n += len(xb)
    return total / n

hist = {"train": [], "val": []}
for epoch in range(100):
    hist["train"].append(run_epoch(train_dl, True))
    hist["val"].append(run_epoch(val_dl, False))
    if epoch % 10 == 0:
        print(f"epoch {epoch:3d}  train {hist['train'][-1]:.4f}  val {hist['val'][-1]:.4f}")

plt.plot(hist["train"], label="train")
plt.plot(hist["val"], label="val")
plt.xlabel("epoch"); plt.ylabel("cross-entropy loss"); plt.legend()
plt.title("MLP on make_moons")
plt.savefig("loss_curve.png", dpi=150)

# TODO: 검증 정확도 계산, 결정 경계 시각화(meshgrid + contourf)
```

### Day 6 실험 3종

| 실험 | 바꿀 설정 | 관찰 포인트 |
| --- | --- | --- |
| 학습률 | `lr` = 1e-1, 1e-3, 1e-5 | 발산 / 정상 수렴 / 거의 줄지 않음을 Loss 곡선 한 장에 겹쳐 그리기 |
| 과적합 | `n_samples=100`, `d_hidden=256`, 500 epoch | 학습 Loss는 계속 줄지만 검증 Loss가 다시 오르는 시점 찾기 |
| 디바이스 | `device` = cpu vs mps, epoch당 시간 측정 | 작은 모델에서는 데이터 전송 비용 때문에 CPU가 더 빠를 수 있음을 확인 |

확장 과제(시간이 남으면): 같은 코드를 torchvision의 FashionMNIST(784차원 입력)로 바꿔 MPS의 이점이 커지는지 확인한다.

## 시스템 트랙 — MLX-LM으로 첫 로컬 LLM 실행 (Day 7)

Day 7에는 이미 학습된 LLM을 Mac에서 직접 돌려, 앞으로 4주 동안 만들 Tokenizer·Attention·Decoder가 최종적으로 어떤 결과를 내는지 미리 체감한다. MLX-LM의 기본 모델은 `mlx-community/Llama-3.2-3B-Instruct-4bit`이며, `--model`로 MLX Community의 다른 모델을 지정할 수 있다 ([MLX-LM README](https://github.com/ml-explore/mlx-lm)).

**1) CLI로 실행**

```bash
# 기본 모델(3B, 4bit)로 한 번 생성 — 첫 실행 시 Hugging Face에서 모델을 내려받음
mlx_lm.generate --prompt "Explain backpropagation in three sentences."

# 대화형 REPL
mlx_lm.chat

# 더 작은 모델로 비교 (모델 이름은 huggingface.co/mlx-community에서 확인)
mlx_lm.generate --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    --prompt "Explain backpropagation in three sentences." --max-tokens 200
```

**2) Python API로 실행하고 수치 기록**

```python
# 05-inference/mlx/first_run.py
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
messages = [{"role": "user", "content": "Explain backpropagation in three sentences."}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
text = generate(model, tokenizer, prompt=prompt, max_tokens=200, verbose=True)
# verbose=True면 프롬프트·생성 토큰 수, tokens-per-sec, 최대 메모리가 함께 출력됨
```

**3) 기록할 것** — `05-inference/mlx/mlx_first_run.md`에 아래 표를 채운다. README 2단계·4단계의 평가 지표(Tokens/sec, Peak Memory)를 미리 측정해 보는 연습이다.

| 모델 | 프롬프트 tokens/sec | 생성 tokens/sec | Peak memory (GB) | 답변 품질 메모 |
| --- | --- | --- | --- | --- |
| Llama-3.2-3B-Instruct-4bit |  |  |  |  |
| 0.5B급 소형 모델 |  |  |  |  |

**4) 연결 질문** — 답을 회고에 적는다.

- `tokenizer.apply_chat_template`가 만든 `prompt`를 출력해 보면 어떤 특수 토큰이 붙는가? (2주차 Tokenization과 연결)
- 생성 속도는 왜 프롬프트 처리 속도보다 느린가? (15주차 Prefill/Decode와 연결)

## 완료 기준과 자가 점검

아래 체크리스트를 모두 통과하면 2주차(Tokenization, BPE, Embedding)로 넘어간다. README의 1주차 완료 기준인 '학습·검증 Loss 시각화'는 마지막 두 항목에 해당한다.

**완료 체크리스트**

- [ ] 환경 검증 스크립트 통과 (arm64, MPS True, MLX GPU)
- [ ] `mini_autograd.py`의 기울기가 수치미분과 소수점 5자리 이내로 일치
- [ ] 같은 식에서 `mini_autograd`와 `torch.autograd`의 기울기가 일치
- [ ] 교재 코드를 보지 않고 학습 루프(forward → loss → zero_grad → backward → step)를 작성
- [ ] `loss_curve.png`에 학습·검증 Loss가 함께 그려지고, 검증 정확도 95% 이상
- [ ] Day 6 실험 3종의 결과를 표와 그래프로 정리
- [ ] MLX-LM으로 모델 1개 이상 실행하고 tokens/sec·메모리 기록

**자가 점검 질문** — 남에게 설명하듯 2~3문장으로 답할 수 있어야 한다.

1. `view`와 `reshape`는 언제 다르게 동작하는가?
2. 역전파에서 기울기를 `=`가 아니라 `+=`로 누적하는 이유는?
3. `optimizer.zero_grad()`를 빼먹으면 어떤 일이 생기는가?
4. 검증 단계에서 `torch.no_grad()`와 `model.eval()`은 각각 무엇을 바꾸는가?
5. 학습 Loss는 줄고 검증 Loss는 오를 때 무엇을 의심하고, 어떤 조치를 취하는가?
6. 작은 MLP에서 MPS가 CPU보다 느릴 수 있는 이유는?
7. Cross-Entropy Loss는 Softmax와 어떤 관계인가? (5~8주차 언어모델 학습에서 그대로 사용)

## 결과물 디렉터리와 기록 방법

1주차 결과물은 README 4.4의 24주 공통 구조 중 `01-pytorch/`와 `05-inference/mlx/`에 둔다. 저장소의 `week-01/` 폴더에는 회고와 결과 요약만 링크해 두면 주차별 진행을 한눈에 볼 수 있다.

```
llm-study/
├── requirements.txt
├── NOTES.md                     # 매일 3줄 기록
├── 01-pytorch/
│   ├── check_env.py
│   ├── tensor/
│   │   ├── tensor_basics.ipynb
│   │   └── numpy_linreg.py
│   ├── autograd/
│   │   ├── mini_autograd.py
│   │   └── autograd_check.py
│   └── neural-network/
│       ├── mlp_train.py
│       ├── loss_curve.png
│       └── experiments.md       # Day 6 실험 3종 결과
├── 05-inference/
│   └── mlx/
│       ├── first_run.py
│       └── mlx_first_run.md
└── week-01/
    └── week01_retro.md
```

**회고 템플릿 (`week01_retro.md`)**

```markdown
# Week 01 회고
- 실제 투입 시간: __시간 (계획 11.5시간)
- 완료 체크리스트: _/7
- 가장 잘 이해한 개념:
- 아직 흐릿한 개념:
- 막혔던 문제와 해결 방법:
- 2주차에 가져갈 질문:
```

**커밋 규칙** — `week01: day3 mini autograd tanh backward`처럼 주차·일차·내용을 남기면, 24주 후 학습 이력이 그대로 포트폴리오가 된다.

## 교재 읽기 범위와 참고 자료

1주차 읽기는 두 교재의 PyTorch·역전파 부분으로 한정하고 아래처럼 나눠 읽는 것을 제안한다.

| 자료 | 1주차 범위 | 언제 |
| --- | --- | --- |
| 『밑바닥부터 시작하는 딥러닝』 | 4장(신경망 학습·수치미분), 5장(오차역전파법). 1~3장은 아는 내용이면 훑어보기 | Day 2~3 |
| 『밑바닥부터 만들면서 배우는 LLM』 / [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) | 부록 A (PyTorch 소개: Tensor, Autograd, 신경망, 학습 루프) | Day 1, 4, 5 |
| PyTorch 공식 튜토리얼 | Autograd 기초, nn.Module | Day 4~5 보충 |
| [MLX-LM](https://github.com/ml-explore/mlx-lm) | README의 Quick Start, Python API | Day 7 |
| [MLX Framework](https://ml-explore.github.io/mlx/) | 설치 문서 (문제 발생 시) | Day 1 |

**Sources**

- [ml-explore/mlx-lm — README](https://github.com/ml-explore/mlx-lm) (CLI·Python API, 기본 모델)
