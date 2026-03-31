import zipfile
import os
from pathlib import Path
import itertools

zip_path = 'imagenet-object-localization-challenge.zip'
base_out_dir = Path('./data/imagenet')

# Tworzymy docelowe foldery
(base_out_dir / 'train').mkdir(parents=True, exist_ok=True)
(base_out_dir / 'val').mkdir(parents=True, exist_ok=True)

print("Otwieram archiwum (to może potrwać kilkanaście sekund)...")

with zipfile.ZipFile(zip_path, 'r') as z:
    members = z.namelist()
    
    # Filtrujemy chirurgicznie: tylko to, co jest w train i val
    files_to_extract = [
        m for m in members 
        if m.startswith('ILSVRC/Data/CLS-LOC/train/') or m.startswith('ILSVRC/Data/CLS-LOC/val/')
    ]
    
    print(f"Znaleziono {len(files_to_extract)} plików do wypakowania. Zaczynamy mielenie...")
    
    for i, member in enumerate(files_to_extract):
        # Ucinamy długi prefiks z Kaggle, żeby zapisać prosto do ./data/imagenet/
        rel_path = member.replace('ILSVRC/Data/CLS-LOC/', '')
        target_path = base_out_dir / rel_path
        
        # Jeśli ścieżka to folder, po prostu go tworzymy
        if member.endswith('/'):
            target_path.mkdir(parents=True, exist_ok=True)
            continue
            
        # Upewniamy się, że folder nadrzędny (np. nazwa klasy) istnieje
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Zapisujemy fizyczny plik .JPEG
        with z.open(member) as source, open(target_path, "wb") as target:
            target.write(source.read())
            
        # Prosty pasek postępu, żebyś wiedział, że skrypt nie umarł
        if (i + 1) % 10000 == 0:
            print(f"Wypakowano {i + 1} z {len(files_to_extract)} plików...")

print("Wypakowywanie zakończone sukcesem!")
