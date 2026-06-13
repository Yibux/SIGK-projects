import os
import pandas as pd
from config import DATA_DIR, DATA_METADATA_DIR, MOTION_EXCEL, DESCRIPTION_EXCEL

def create_local_index():
    print("Skanowanie folderów w poszukiwaniu plików .bvh...")
    
    available_bvh = {}
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith('.bvh'):
                available_bvh[file] = os.path.join(root, file)
                
    print(f"Fizycznie znaleziono {len(available_bvh)} plików .bvh na dysku.")
    print("Analizowanie pliku z Excela...")
    
    clean_records = []
    
    excel_path = os.path.join(DATA_METADATA_DIR, 'cmu-mocap-index-spreadsheet.xls')

    try:
        df = pd.read_excel(excel_path, skiprows=10)
    except Exception as e:
        print(f"Błąd odczytu Excela: {e}")
        print("Jeśli brakuje Ci biblioteki xlrd, wpisz w konsoli: pip install xlrd")
        return
        
    for index, row in df.iterrows():
        if pd.isna(row.get(DESCRIPTION_EXCEL)):
            continue
            
        desc = str(row[DESCRIPTION_EXCEL]).lower()
        if str(desc).__contains__('walk') or str(desc).__contains__('jump'):
            
            label = 'walk' if str(desc).__contains__('walk') else 'jump'
            motion = row.get(MOTION_EXCEL)
            try:                
                filename = f"{motion}.bvh"
                label = 'walk' if 'walk' in desc else 'jump'
                
                if filename in available_bvh:
                    clean_records.append({
                        'filename': filename, 
                        'filepath': available_bvh[filename],
                        'action': label
                    })
                    
            except ValueError:
                continue

    if len(clean_records) > 0:
        clean_df = pd.DataFrame(clean_records)
        output_path = os.path.join(DATA_DIR, 'filtered_dataset.csv')
        clean_df.to_csv(output_path, index=False)
        print(f"Gotowe! Utworzono '{output_path}'. Powiązano {len(clean_records)} plików.")
    else:
        print("Nie udało się powiązać żadnych plików. Upewnij się, że nazwy plików to np. '01_01.bvh'.")

if __name__ == "__main__":
    create_local_index()