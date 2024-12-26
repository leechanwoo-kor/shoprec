# ShopRec

**프로젝트 개요**

- 본 프로젝트는 쇼핑몰 추천 시스템 구축을 목표로 합니다.

**디렉토리 구조**

- data/
  - raw/, processed/, ...
- src/
  - data/, models/, training/, inference/, evaluation/, utils/ ...
- notebooks/
- scripts/
- tests/
- config/

**개발 환경**

- Python 3.11
- PyTorch 2.5.1
- 기타 주요 라이브러리는 `requirements.txt` 참고

**사용 방법**

1. `pip install -r requirements.txt` (또는 `conda env create -f environment.yml`)
2. `bash scripts/run_preprocess.sh` : 데이터 전처리
3. `bash scripts/run_train.sh` : 모델 학습
4. `bash scripts/run_infer.sh` : 배치 추론 결과 생성

- set PYTHONPATH=. && python