## HuggingFace gguf 파일을 Ollama 로딩

> HuggingFace-Hub 설치
```bash
pip install huggingface-hub
```

아래의 예시는 `EEVE-Korean-Instruct-10.8B-v1.0`
- HF: https://huggingface.co/yanolja/EEVE-Korean-Instruct-10.8B-v1.0
- GGUF: https://huggingface.co/heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF

GGUF 파일을 다운로드 받기 위하여 https://huggingface.co/heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF 에서 원하는 .gguf 모델을 다운로드 받습니다.

순서대로
- `HuggingFace Repo`s
- .gguf 파일명
- local-dir 설정
- 심볼릭 링크 설정
  
```bash
huggingface-cli download \
  unsloth/gemma-4-E4B-it-GGUF \
  gemma-4-E4B-it-Q4_K_M.gguf \
  --local-dir C:\Users\User\Desktop\Projects\ollama_model\
  --local-dir-use-symlinks False
```

### Modelfile

> EEVE-Korean-Instruct-10.8B-v1.0 예시
```
FROM ggml-model-Q5_K_M.gguf

TEMPLATE """{{- if .System }}
<s>{{ .System }}</s>
{{- end }}
<s>Human:
{{ .Prompt }}</s>
<s>Assistant:
"""

SYSTEM """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."""

PARAMETER stop <s>
PARAMETER stop </s>
```

> openbuddy-llama2-13b 예시
```
FROM openbuddy-llama2-13b-v11.1.Q4_K_M.gguf

TEMPLATE """{{- if .System }}
<|im_start|>system {{ .System }}<|im_end|>
{{- end }}
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

SYSTEM """"""

PARAMETER stop <|im_start|>
PARAMETER stop <|im_end|>
```

## Ollama 실행

```bash
ollama create EEVE-Korean-10.8B -f EEVE-Korean-Instruct-10.8B-v1.0-GGUF/Modelfile
```

Ollama 모델 목록

```bash
ollama list
```

Ollama 모델 실행

```bash
ollama run EEVE-Korean-10.8B:latest
```

## LangServe 에서 Ollama 체인 생성

app 폴더 진입 후

```bash
python server.py
python fastapi_server.py
```

## ngrok 에서 터널링(포트 포워드)

```bash
ngrok http localhost:8000
```
![](./images/capture-20240411-035817.png)

NGROK 도메인 등록 링크: https://dashboard.ngrok.com/cloud-edge/domains

> 고정 도메인이 있는 경우
```bash
ngrok http --domain=poodle-deep-marmot.ngrok-free.app 8000
```

| 타입           | 설명            |
| -------------- | --------------- |
| **[Feat]**  | 새로운 기능 추가 |
| **[Enhancement]** | 기존 기능 개선 |
| **[Doc]** | 문서화 관련 |
| **[Task]** | 특정 작업 또는 할 일 |
| **[Refactor]** | 코드 리팩토링 |
| **[Fix]** | 버그 수정 |
| **[!HOTFIX]** | 급한 치명적인 버그 수정 |
| **[Chore]** | 빌드 업무 수정, 패키지 매니저 수정 |
| **[Style]** | 코드 포맷팅, 코드 오타, 함수명 수정 등 |
| **[UI]** | XML 화면 설계 |
| **[Test]** | 테스트 관련 |
| **[Bug]** | 버그 관련 이슈 |

-----

### 1. issue convention

**📌 형식**:

[타입] 이슈 내용


**📌 예시**:

[Feat] User 도메인 구현<br>
[Refactor] User 관련 DTO 수정

⭐️ assigner와 해당하는 라벨도 체크해주세요!

  -------------

### 2. branch convention

**📌 형식**:

타입/#이슈번호-간단한 설명

⭐️ 이때 타입은 해당 Branch의 이슈 타입과 동일하게 가져가시면 됩니다!


**📌 예시**:

feat/#1-User-도메인-설계<br>
refactor/#32-쿼리-최적화

----------------

### 3. commit convention

**📌 형식**:

[커밋 타입/#이슈번호] : 커밋 내용<br>

**📌 예시**:

[feat/#32] : User 도메인 구현<br>
[feat/#32] : User 필드값 annotation 추가<br>

