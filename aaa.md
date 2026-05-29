중간고사: 노션, 갤럭시 노트, 드라이브 확인

# 가상환경 활성화
### powershell
    venv\Scripts\Activate.ps1
### mac
    .\venv\Scripts\Activate
### git bash
    source venv/Scripts/activate

# 파일 스테이징 및 커밋
git add .
git commit -m "프로젝트 초기 설정"

# 원격 저장소 연결
git remote add origin https://github.com/shimyoubin/2026_bigdata_QB.git

# 브랜치 이름 변경 및 Push
git branch -M main
git push -u origin main

# 스트림릿 app.py 실행
streamlit run app.py

| 명령어 | 설명 |
|--------|------|
| `git status` | 현재 변경 상태 확인 |
| `git log --oneline` | 커밋 이력 간략 조회 |
| `git remote -v` | 연결된 원격 저장소 확인 |
| `git remote remove origin` | 원격 저장소 연결 해제 |

pip install -r requirements.txt

# 초기 설정
git config --global user.email "shimub05@naver.com"
git config --global user.name "Youbin"

# JUPYTER LAB 실행
jupyter lab