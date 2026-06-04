import json
from pathlib import Path

def path_to_dict(path, exclude_names):
    path = Path(path)

    # Si el nombre está en la lista de exclusión, retornamos None
    if path.name in exclude_names:
        return None

    d = {'name': path.name}
    if path.is_dir():
        d['type'] = 'directory'
        # Filtramos los None resultantes de archivos omitidos
        children = [path_to_dict(p, exclude_names) for p in path.iterdir()]
        d['children'] = [c for c in children if c is not None]
    else:
        d['type'] = 'file'
        d['size_bytes'] = path.stat().st_size
    return d

def save_structure_to_json(root_path, output_file, exclude_names):
    structure = path_to_dict(root_path, exclude_names)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(structure, f, indent=4, ensure_ascii=False)

    print(f"JSON generado omitiendo: {', '.join(exclude_names)}")

if __name__ == "__main__":
    # Lista de archivos o carpetas a ignorar
    omitir = {".rhistory", ".git", "__pycache__", ".DS_Store"}

    directorio_a_escanear = "C:/Users/equih/0 Versiones/workflow-iie/propuesta-workflow-iie"
    archivo_salida = "estructura_proyecto.json"

    save_structure_to_json(directorio_a_escanear, archivo_salida, omitir)