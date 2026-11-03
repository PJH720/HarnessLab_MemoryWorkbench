# Project101: 메모리를 활용하는 에이전트 평가

[![Presentation](https://img.shields.io/badge/Presentation-Vercel-black?logo=vercel)](https://harness-midterm-talk.vercel.app)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)

[**English Documentation**](README.md) · [**웹 발표 슬라이드 (Vercel)**](https://harness-midterm-talk.vercel.app) · [**전체 분석 그림 갤러리**](analysis/figures/README.md)

> [!IMPORTANT]
> **포지셔닝 — 고정된 언어 모델에서 메모리 전략의 효과를 조사한 개인 파일럿 연구와 인터랙티브 실험 도구**
>
> **연구 질문:** 언어 모델(`Qwen3.8-Flash-Next-NVFP4`, reasoning 비활성)을 고정한 채 프롬프트에 주입하는 기억(*하네스*)만 바꾸면 정확도가 달라지는가? 그리고 모델이 스스로 요약·생성한 진단 단서(`cue_bank`)가 단순 예시 검색(`knn_top6`)보다 뛰어난가?
>
> **3줄 핵심 결론:**
> 1. **검색 기억은 무메모리를 압도한다** (`knn_top6` **83.0%** vs `no_memory` **67.0%**, 쌍체 부트스트랩 $\Delta = +16.0\text{\%p}$, seed 42 정확 McNemar $p = 5.6 \times 10^{-6}$).
> 2. **상위 6개 검색만으로 전체 60개 예시 주입과 동등한 성능을 내며 토큰은 80.8% 절감한다** (`knn_top6` **83.0%** vs `fewshot_all` **84.6%**, $p = 0.18$, **입력 토큰 약 1/5**).
> 3. **LLM이 생성한 단서는 단순 검색을 이기지 못했다** (`cue_bank` $-0.8\text{\%p}$, 95% CI $[-3.3, +1.7]$). 소표본 validation(50문항)에서 보였던 $12\text{\%p}$ 격차는 본 평가인 test(212문항)에서 사라졌다($p = 0.68$).

---

## 빠른 탐색 (목차)

1. [한눈에 보는 핵심 결과 & 시각적 증거 (Fig 3, Fig 1)](#1-한눈에-보는-핵심-결과--시각적-증거-fig-3-fig-1)
2. [왜 소표본 Validation에 속는가: 표본 크기와 시드 변동 (Fig 4, Fig 2)](#2-왜-소표본-validation에-속는가-표본-크기와-시드-변동-fig-4-fig-2)
3. [왜 생성 단서(Cue)는 실패했는가: 오답 쏠림과 회귀 분석 (Fig 5)](#3-왜-생성-단서cue는-실패했는가-오답-쏠림과-회귀-분석-fig-5)
4. [통계적 엄밀성: 검정력 곡선, 추론 잡음 및 연구의 한계 (Fig 7, Fig 8)](#4-통계적-엄밀성-검정력-곡선-추론-잡음-및-연구의-한계-fig-7-fig-8)
5. [도메인 확장 파일럿: 데이터사이언스 에이전트 (Fig 6)](#5-도메인-확장-파일럿-데이터사이언스-에이전트-fig-6)
6. [인터랙티브 실험 및 학습 도구 (`harness-cli`)](#6-인터랙티브-실험-및-학습-도구-harness-cli)
7. [저장소 구조, 테스트 재현 및 개인정보 보호](#7-저장소-구조-테스트-재현-및-개인정보-보호)

---

## 1. 한눈에 보는 핵심 결과 & 시각적 증거 (Fig 3, Fig 1)

이 파일럿 연구는 **Symptom2Disease**(22개 질병 라벨, 시드 3개: `42`, `123`, `456`)에서 7종의 메모리 하네스를 비교했습니다. 질병 예측은 벤치마크 텍스트 분류이며 임상 진단이 아닙니다.

### 시각 자료 1A — `knn_top6` 대비 시스템별 성능 차이 (쌍체 부트스트랩 95% CI)

![knn_top6 대비 각 시스템의 쌍체 부트스트랩 95% CI (test, 시드당 n=212)](analysis/figures/fig3_bootstrap_forest_plot.png)

* **읽는 법:** 세로 점선(`0`)이 기준선인 `knn_top6`(유사 예시 6개 검색)입니다. 95% 신뢰구간이 `0`에서 확실히 벗어난 것은 `no_memory`($-16.0\text{\%p}$, CI $[-19.8, -12.4]$)뿐이며, 생성 단서 계열(`cue_bank`, `cue_bank_verified`, `reflexion_notes`)은 모두 `0`을 가로지릅니다.

### 시각 자료 1B — 정확도 vs 입력 토큰 파레토 프론티어 (Pareto Frontier)

![정확도와 입력 토큰의 파레토 관계](analysis/figures/fig1_accuracy_vs_tokens_pareto.png)

* **읽는 법:** 학습 예시 60개를 매번 프롬프트에 전부 넣는 `fewshot_all`(test 시드 평균 607,449 토큰, **84.6%**)과 비교할 때, TF-IDF로 유사 예시 6개만 골라 넣는 `knn_top6`(116,673 토큰, **83.0%**)는 **입력 토큰을 80.8% 절감**하면서도 통계적으로 구별되지 않는 성능($+1.6\text{\%p}$, 95% CI $[-0.8, +3.8]$, pooled Mantel-Haenszel $p = 0.18$)을 달성합니다.

### 핵심 결과 요약표 (Test 분할, 시드당 $n = 212$)

| 핵심 발견 | 정량적 근거 (Seeds 42 / 123 / 456) |
|---|---|
| **1. 검색 기억은 기억 없음보다 확실히 낫다** | `knn_top6` **83.0%** vs `no_memory` **67.0%** (3-seed 평균); 부트스트랩 차이 $-16.0\text{\%p}$ $[-19.8, -12.4]$; seed 42 정확 McNemar $p = 5.6 \times 10^{-6}$ (Holm $p = 3.3 \times 10^{-5}$) |
| **2. 예시 6개 검색이 60개 전체 주입과 대등하다** | `fewshot_all` **84.6%** vs `knn_top6` **83.0%**: $+1.6\text{\%p}$, CI $[-0.8, +3.8]$, MH $p = 0.18$; **`knn_top6` 입력 토큰 80.8% 절감** |
| **3. 생성 단서는 단순 검색을 넘지 못한다** | `cue_bank` $-0.8\text{\%p}$ $[-3.3, +1.7]$; `cue_bank_verified` $+0.6\text{\%p}$ $[-1.4, +2.7]$; `reflexion_notes` $-0.5\text{\%p}$ $[-2.7, +1.7]$ |
| **4. 소표본(n=50) 격차는 재현되지 않았다** | Seed 42 validation($n=50$): `knn_top6` 41/50 vs `cue_bank` 35/50 ($p = 0.070$) $\rightarrow$ Test($n=212$): 175/212 vs 172/212 ($p = 0.68$) |
| **5. 단서 오답은 주입된 단서를 따라간다** | 정답 라벨이 단서에 포함됐음에도 틀린 `cue_bank` 오답 45건 중 **34건(76%)**이 단서에 적힌 다른 질병을 예측함 |
| **6. Temperature 0에서도 추론은 흔들린다** | 동일한 요청을 반복 호출할 때 validation 문항의 **12~14%**에서 예측 라벨이, **6~8%**에서 정오(맞고 틀림)가 바뀜 |

---

## 2. 왜 소표본 Validation에 속는가: 표본 크기와 시드 변동 (Fig 4, Fig 2)

초기 소표본(50문항) 관찰을 사전 등록한 후속 본 평가(212문항)로 재검증하면서 확인한 핵심 교훈입니다.

| 소표본(n=50) vs 본 평가(n=212) 재현성 (`Fig 4`) | 분할 및 시드별 성능 변동 폭 (`Fig 2`) |
|---|---|
| ![소표본과 대표본의 재현성](analysis/figures/fig4_sample_size_reproducibility.png) | ![분할별 성능과 시드 변동](analysis/figures/fig2_system_performance_by_split.png) |
| **왼쪽 (`Fig 4`):** Seed 42의 동일한 동결 메모리에서 `cue_bank - knn_top6` 격차는 validation(50문항)에서 **$-12.0\text{\%p}$**(불일치 문항 7:1, $p=0.070$)로 크게 보였으나, test(212문항)에서는 **$-1.4\text{\%p}$**(불일치 13:10, $p=0.678$)로 수렴했습니다. | **오른쪽 (`Fig 2`):** 50문항에서는 단 한 문항이 $2.0\text{\%p}$에 해당하므로 시드 간 최대-최소 폭이 validation에서 최대 **10.0%p**에 달했으나, 212문항 test에서는 최대 **3.3%p**로 줄어들었습니다. |

---

## 3. 왜 생성 단서(Cue)는 실패했는가: 오답 쏠림과 회귀 분석 (Fig 5)

LLM이 요약한 진단 규칙(`cue_bank`, `cue_bank_verified`, `reflexion_notes`)이 실제 추론에서 어떻게 작동하는지 분석했습니다.

![단서 기억의 인지적 닻 피처](analysis/figures/fig5_cognitive_anchor_features.png)

* **오답의 단서 쏠림 현상 (오른쪽 히트맵):** 정답 질환이 주입된 단서 안에 들어 있었음에도 틀린 `cue_bank` 오답 45건 중 **76%(34/45)**는 *단서에 등장한 다른 질환*으로 답이 끌려갔습니다.
* **사전 등록 군집 로지스틱 회귀 (가운데 패널, H3):** 문항 단위 군집 강건 표준오차(clustered SE)를 적용한 회귀 분석에서 `target_in_cues`(단서 내 정답 포함 여부) 계수는 $-0.044$ ($p = 0.817$)로 유의하지 않았습니다. 실제 오답률을 결정지은 유의한 변수는 **검색된 예시와의 최대 유사도**(`max_similarity` $\beta = -2.921, p = 0.014$)와 **프롬프트 길이**(`prompt_kchars` $\beta = -1.488, p = 0.003$)였습니다. 즉, 서술적인 오답 쏠림 현상만으로 Negative Priming의 인과적 효과를 단정할 수 없습니다.

---

## 4. 통계적 엄밀성: 검정력 곡선, 추론 잡음 및 연구의 한계 (Fig 7, Fig 8)

| 정확 McNemar 통계적 검정력 곡선 (`Fig 7`) | 로컬 LLM 서빙의 반복 호출 잡음 (`Fig 8`) |
|---|---|
| ![통계적 검정력 곡선](analysis/figures/fig7_power_analysis_curve.png) | ![로컬 LLM 서빙의 반복 호출 잡음](analysis/figures/fig8_runtime_stochasticity.png) |
| **왼쪽 (`Fig 7`):** 불일치율 20%, 양측 $\alpha = 0.05$ 기준에서 $5\text{\%p}$ 차이를 검정력 80%로 잡아내려면 **독립 쌍체 문항 $n = 658$개**가 필요합니다. $n = 50$에서의 검정력은 **0.07**, $n = 212$에서는 **0.32**에 불과합니다. | **오른쪽 (`Fig 8`):** 동일 입력·동일 설정(`temperature=0`)으로 재호출한 세 쌍(val 50문항)에서 예측 라벨은 **12~14%(6~7문항)**, 정오는 **6~8%(3~4문항)** 바뀌었습니다. 소표본의 몇 문항 차이는 이 서빙 잡음 범위 안에 있습니다. |

> [!WARNING]
> ### 이 결과가 보여 주지 않는 것 (명시적 한계)
> - **사전 등록 주가설의 3-시드 판정은 `INCOMPLETE`(미완)입니다:** 사전 등록한 필수 결과 파일 37개 중 **29개**만 분석 가능했고 `seed 456` 파일 8개가 누락되었습니다. 시드 3개 미만에서는 최종 결론을 내리지 않는다는 사전 규칙을 엄격히 적용합니다.
> - **비유의성($p \ge 0.05$)은 동등성의 증명이 아닙니다:** $n = 212$는 $5\text{\%p}$ 내외의 미세한 차이를 판별하기에 검정력(0.32)이 부족합니다.
> - **반복 평가는 독립 표본이 아닙니다:** 세 시드는 동일한 212개 test 문항을 재평가한 것이므로 총 636회 평가가 독립 문항 636개를 의미하지 않습니다.
> - **단일 벤치마크·단일 동결 모델:** 본 결과는 `Symptom2Disease`와 `Qwen3.8-Flash-Next-NVFP4`에 한정되며 임상적 유효성을 주장하지 않습니다.

<details>
<summary><strong>📊 클릭하여 펼치기: 검증된 과거 스냅샷 상세 표 (Exact McNemar & Holm 보정 p-value)</strong></summary>

아래 표는 `2026-11-02 02:32 KST`에 저장된 로컬 분석 스냅샷입니다.

| 비교 항목 | 정답 수 | 정확 McNemar $p$ | Holm 보정 $p$ | 통계적 해석 |
|---|---|---:|---:|---|
| **seed 42 test:** `knn_top6` vs `no_memory` | 175/212 vs 142/212 | `0.00000555` | `0.0000333` | **유의한 보조 비교** ($+15.57\text{\%p}$) |
| **seed 42 test:** `knn_top6` vs `cue_bank` | 175/212 vs 172/212 | `0.6776` | `1.0000` | 유의한 차이를 발견하지 못함 |
| **seed 123 test:** `knn_top6` vs `cue_bank` | 173/212 vs 179/212 | `0.2101` | `0.8405` | 유의한 차이 없음 (방향 반대) |
| **seed 42 val:** `knn_top6` vs `cue_bank` | 41/50 vs 35/50 | `0.0703` | `0.4219` | 다중 비교 보정 후 유의하지 않음 ($n=50$) |

- **검증 일치도:** 분석·검정력 단위 테스트 26개 통과, 기준 비교 21개 및 문항 행 350개 바이트 단위 일치 확인.
- **실험 결과 파일 없이 독립 실행 가능한 검정력 계산기:**
  ```sh
  mkdir -p analysis/multiseed/out
  uv run --project harness_lab/text_classification --with scipy \
    python analysis/multiseed/power_plan.py
  ```

</details>

---

## 5. 도메인 확장 파일럿: 데이터사이언스 에이전트 (`Fig 6`)

단일 턴 분류를 넘어 다단계 도구 호출(ReAct) 과제에서도 동일한 현상이 나타나는지 확인하기 위해, 데이터사이언스 과제 카드(`ds_tasks`)에서 단일 시드 파일럿을 수행했습니다.

![데이터사이언스 에이전트 파일럿](analysis/figures/fig6_ds_tasks_benchmark.png)

* **처음 보는 평가 과제 (`eval`, 하네스당 $n = 31$, 학습 단계 제외):**
  * `ds_knn` (유사 궤적 검색): 정확도 **80.6%** ($25/31$), 중앙 스텝 **2회**, 중앙 입력 토큰 **4,334개**.
  * `ds_base` (기억 없음): 정확도 **48.4%** ($15/31$), 중앙 스텝 **7회**, 중앙 입력 토큰 **16,042개**.
  * `ds_cue_verified` (검증된 규칙 주입): 정확도 **38.7%** ($12/31$).
* **해석상 주의점:** 시드 1개 파일럿이며, 검색된 `train` 궤적이 `eval` 정답과 얼마나 유사한지 별도 감사를 수행하기 전이므로 확정 결론이 아닌 탐색적 관찰로 보고합니다.

---

## 6. 인터랙티브 실험 및 학습 도구 (`harness-cli`)

`harness_cli` v0.2.2는 위 연구 결과를 터미널에서 직접 체험하고 일상 학습에도 활용할 수 있도록 **두 가지 모드를 완전히 분리**하여 제공합니다.

| 모드 구분 | 핵심 용도 | 실행 명령어 |
|---|---|---|
| **1. 실용 학습 어시스턴트 (`study`, 기본값)** | 내 강의 노트(`.md`/`.txt`)를 로드해 TF-IDF/KNN으로 질문하고, 오답 교훈을 `~/.harness`에 영구 축적합니다. 시작 시 질병 프리셋이 전혀 노출되지 않습니다. | `uv run --project harness_cli harness-cli` <br> `uv run --project harness_cli harness-cli ask "What is L2 regularization?" --doc notes.md` |
| **2. 캡스톤 연구 재현 랩 (`medical`, 격리됨)** | 연구에서 평가한 8개 동결 메모리 블록을 레고처럼 교체하며 실제 주입 프롬프트(`/inspect context`)와 블록 간 성능(`/compare`)을 대조합니다. | `uv run --project harness_cli harness-cli --demo compare no_memory knn_top6 1` <br> 또는 REPL 내 `/task medical` 입력 |

### 30초 빠른 시작 (모델 서버 없이 즉시 실행)

```sh
# 1. CLI 의존성 설치
uv sync --project harness_cli --group dev

# 2. 기본 실용 학습 모드로 대화형 REPL 시작
uv run --project harness_cli harness-cli

# 3. 오프라인 녹화 캐시(--demo)로 연구 핵심 비교 즉시 재현
uv run --project harness_cli harness-cli --demo compare no_memory knn_top6 1
```

<details>
<summary><strong>🛠️ 클릭하여 펼치기: 실시간 LLM 엔드포인트 연결, 대화형 명령어 전체 목록 & 독립 실행 번들</strong></summary>

#### 실시간 모델 서버 연결 (Ollama / OpenAI / vLLM)
새 컴퓨터에서 엔드포인트를 한 번 지정하면 `~/.harness/config.json`에 저장됩니다(API 키는 파일에 저장하지 않고 환경 변수나 `.env`에서 읽습니다):

```sh
# 로컬 Ollama 연결 예시
uv run --project harness_cli harness-cli config --api-base http://localhost:11434/v1 --model llama3.2:3b

# OpenAI API 연결 예시
OPENAI_API_KEY=... uv run --project harness_cli harness-cli config --api-base https://api.openai.com/v1 --model gpt-4o-mini
```

#### 대화형 REPL 주요 명령어

| 명령어 | 동작 설명 |
|---|---|
| `/load <notes.md\|notes.txt>` | 내 마크다운/텍스트 노트를 제목·문단 단위로 색인하고 `study` 모드로 전환 |
| `/record "<개념>" -> "<규칙>"` | 학습 규칙을 영구 기억에 가르침 (`medical` 모드에서는 증상-질병 예시 추가) |
| `/harness <이름>` (별칭 `/switch`) | 메모리 전략 교체 (`study`: `none`, `knn`, `cues` / `medical`: 8대 동결 블록) |
| `/compare <A> <B> [질문/1-3]` | 두 메모리 전략을 나란히 실행하여 프롬프트 토큰 수, 검색된 근거, 답변을 좌우 비교 |
| `/inspect context` | 직전 질문의 실제 RAW 프롬프트와 주입된 예시별 TF-IDF 유사도·선택 이유 확인 |
| `/explain fewshot` | `fewshot_all`이 확장 불가능한 이유와 정확도-토큰 파레토 경계 설명 |
| `/profile`, `/quiz [주제]` | 학습자 맞춤 프로필 설정 및 로드된 노트 기반 복습 퀴즈 3문항 생성 |
| `/task medical` / `/task study` | 실용 노트 학습 모드와 22개 질병 연구 재현 랩 간 명시적 전환 |

#### 독립 실행 번들 (Standalone Zip) 빌드
```sh
cd harness_cli && uv build && cd ..
uv run --project harness_cli python scripts/build_standalone.py
# -> dist/harness-cli-v0.2.2-standalone.zip 생성 (wheel + 동결 기억 8종 포함)
```

</details>

---

## 7. 저장소 구조, 테스트 재현 및 개인정보 보호

| 경로 | 역할 |
|---|---|
| [`analysis/figures/`](analysis/figures/README.md) | 8종 고해상도 분석 도표(300 DPI PNG + 벡터 PDF) 및 자동 생성 스크립트 |
| `analysis/multiseed/` | 캐시 전용 다중 시드 통계 엔진, 정확 McNemar, 부트스트랩, 검정력 계획 및 개인정보 가드 |
| `harness_cli/` | 실용 노트 학습 및 8개 블록 벤치마크 재현을 지원하는 Typer/Rich 터미널 앱 |
| `harness_lab/text_classification/` | 7종 온라인 메모리 하네스 구현 및 벤치마크 평가 루프 |
| `ds_tasks/` | 데이터사이언스 과제 카드, ReAct 실행 루프 및 2-fold 교차 적합 평가기 |

### 검증 테스트 실행 명령

```sh
# 1. CLI 단위 및 통합 테스트 실행
uv run --project harness_cli pytest harness_cli/tests -q

# 2. 다중 시드 통계 엔진, 검정력 계산기, 개인정보 보호 가드 테스트 실행
uv run --project harness_lab/text_classification \
  --with scipy --with statsmodels --with pandas --with pytest \
  python -m pytest analysis/multiseed/test_stats.py \
  analysis/multiseed/test_power_plan.py analysis/multiseed/test_privacy_guard.py \
  -k 'not reference_replay_matches_original_features_and_all_pairs' \
  -q -p no:cacheprovider --basetemp analysis/multiseed/.pytest-tmp
```

### 연구 범위, 향후 계획 및 라이선스
- **후속 팀 연구 연계:** 이 저장소는 **동결된 개인 파일럿 연구**입니다. 메모리 계약 검증, 캐시 전용 재생, Holm 보정 쌍체 검정, 검정력 계획($n=658$) 등 검증된 평가 프로토콜은 향후 에이전트 스터디 그룹과 진행할 별도 저장소의 출발점으로 재사용됩니다.
- **개인정보 보호 가드:** 원시 데이터셋, 개인 보고서, 인증 정보, 추론 캐시, 내부 호스트명은 배포에서 제외되었습니다. 클론 후 로컬 커밋/푸시 보호 훅을 켜려면 아래 명령을 실행하세요:
  ```sh
  git config --local core.hooksPath analysis/multiseed/hooks
  ```
- **출처 및 라이선스:** 수정된 Meta-Harness 코드는 [`harness_lab/LICENSE-meta-harness`](harness_lab/LICENSE-meta-harness)의 원본 MIT 고지를 유지하며, 각 데이터셋과 외부 자료는 고유의 라이선스 조건을 따릅니다.
