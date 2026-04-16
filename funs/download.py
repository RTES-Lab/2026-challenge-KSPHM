import gdown
import zipfile
import os
import glob


def download_dataset(root: str):
    # TODO: df 만드는거 해야함
    os.makedirs(root, exist_ok=True)

    if os.path.exists(os.path.join(root, "Train1_Vibration")):
        print("이미 데이터셋이 존재합니다. 다운로드를 건너뜁니다.")
        return

    # 다운로드
    url = "https://drive.google.com/uc?id=1ahTu-haFuwS7orW2Dpt8kZG9Di563sNB"
    output = os.path.join(root, "Train.zip")
    gdown.download(url, output, quiet=False)

    # 압축 해제
    with zipfile.ZipFile(output, 'r') as z:
        z.extractall(root)
        print(f"압축 해제 완료: {len(z.namelist())}개 파일")
    os.remove(output)

    # 내부 zip 파일들도 압축 해제
    for zf in glob.glob(os.path.join(root, "*.zip")):
        with zipfile.ZipFile(zf, 'r') as z:
            z.extractall(root)
        os.remove(zf)
        print(f"{os.path.basename(zf)} 압축 해제 완료")