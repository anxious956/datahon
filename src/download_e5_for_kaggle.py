"""
e5-base'i Kaggle'da internet-KAPALI kullanım için indirme yardımcısı.

KULLANIM (internet AÇIK bir Kaggle notebook'ta bir kez çalıştır):
    python3 src/download_e5_for_kaggle.py
Sonra:
  1. /kaggle/working/multilingual-e5-base klasörünü "Save Version (Save & Run All)" ile output yap.
  2. Output'u "New Dataset" olarak yayınla (örn. ad: multilingual-e5-base).
  3. Asıl Faz 2.2 notebook'unda (internet KAPALI) "Add Input" ile bu dataset'i ekle.
  4. faz2_2_embed.py'yi E5_PATH ile çalıştır:
        E5_PATH=/kaggle/input/multilingual-e5-base/multilingual-e5-base python3 src/faz2_2_embed.py
"""
import os
from huggingface_hub import snapshot_download

OUT = os.environ.get('E5_OUT', '/kaggle/working/multilingual-e5-base')

if __name__ == '__main__':
    path = snapshot_download(repo_id='intfloat/multilingual-e5-base', local_dir=OUT)
    print('İndirildi ->', path)
    print('Dosyalar:', os.listdir(path))
