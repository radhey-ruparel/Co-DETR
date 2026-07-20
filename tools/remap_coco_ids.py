# tools/remap_coco_ids.py
import json, sys, os

if len(sys.argv) < 3:
    print("Usage: python tools/remap_coco_ids.py in.json out.json")
    exit(1)

infile, outfile = sys.argv[1], sys.argv[2]
j = json.load(open(infile))

# --- remap category ids ---
cats = j.get("categories", [])
old_ids = sorted([c['id'] for c in cats])
new_map = {old: new for new, old in enumerate(old_ids, start=1)}

for c in j['categories']:
    c['id'] = new_map[c['id']]

for a in j['annotations']:
    a['category_id'] = new_map[a['category_id']]

# --- fix file_name field ---
for img in j.get("images", []):
    if "file_name" in img:
        clean_path = img["file_name"].replace("\\", "/")
        img["file_name"] = os.path.basename(clean_path)

# --- save ---
with open(outfile, "w") as f:
    json.dump(j, f, indent=2)

print("Remapped ids:", new_map)
print(f"Saved fixed annotations to {outfile}")
