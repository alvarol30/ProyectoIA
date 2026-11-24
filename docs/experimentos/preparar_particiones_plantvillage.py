import csv
import json
from pathlib import Path, PurePosixPath
import sys
from urllib.request import urlopen
from zipfile import ZipFile

from sklearn.model_selection import StratifiedGroupKFold

downloads = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
revision = "9e97599868962bd0079b8db4b7f1efa9185fa1e7"
base = f"https://huggingface.co/datasets/mohanty/PlantVillage/resolve/{revision}"


def get(path):
    with urlopen(f"{base}/{path}", timeout=30) as response:
        return response.read()


def key(name):
    return "/".join(PurePosixPath(name).parts[-2:])


classes = json.loads((downloads / "class_names.json").read_text(encoding="utf-8"))
official = {
    split: {key(line) for line in get(f"splits/color_{split}.txt").decode().splitlines() if line}
    for split in ("train", "test")
}
leaf_map = json.loads(get("leaf_grouping/leaf-map.json"))
rows = {"train": [], "test": []}
known = 0
seen = set()
with ZipFile(downloads / "plantvillage-dataset.zip") as archive:
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if len(path.parts) < 3 or path.parts[-3] != "color" or path.parent.name not in classes:
            continue
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        class_name = path.parent.name
        identity = path.stem.rsplit("___", 1)[-1].split("copy")[0].strip().lower()
        groups = [g for g in leaf_map.get(identity, []) if g.startswith(class_name + ":::")]
        known += int(len(groups) == 1)
        group = groups[0] if len(groups) == 1 else f"{class_name}:::filename::{identity}"
        matches = [split for split, names in official.items() if key(member.filename) in names]
        if len(matches) != 1 or key(member.filename) in seen:
            raise ValueError(f"Asignación ambigua: {member.filename}")
        seen.add(key(member.filename))
        stat = rows[matches[0]]
        stat.append((member.filename, class_name, group))

test_groups = {item[2] for item in rows["test"]}
pool_all = rows["train"]
pool = [item for item in pool_all if item[2] not in test_groups]
labels = [item[1] for item in pool]
groups = [item[2] for item in pool]
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=123)
expected = set(classes)
for train_idx, validation_idx in splitter.split(pool, labels, groups):
    train = [pool[i] for i in train_idx]
    validation = [pool[i] for i in validation_idx]
    if {r[1] for r in train} == expected and {r[1] for r in validation} == expected:
        break
else:
    raise ValueError("No se pudo cubrir las 23 clases en entrenamiento y validación")

splits = {"train": train, "validation": validation, "test": rows["test"]}
for split, entries in splits.items():
    assert {r[1] for r in entries} == expected, split
    print(split, len(entries), len({r[2] for r in entries}))
assert len({r[2] for entries in splits.values() for r in entries}) == sum(
    len({r[2] for r in entries}) for entries in splits.values()
)
print("Coincidencias explícitas con el mapa de hojas:", known)
print("Imágenes del grupo de prueba excluidas del entrenamiento:", len(pool_all) - len(pool))
with (out / "plantvillage_particiones.csv").open("w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(("split", "zip_member", "class_name", "leaf_group"))
    for split, entries in splits.items():
        writer.writerows((split, *row) for row in entries)
