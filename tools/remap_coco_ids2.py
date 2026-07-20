# tools/remap_to_coco_ids.py
import json, sys, os

if len(sys.argv) != 4:
    print("Usage: python tools/remap_to_coco_ids.py in.json out.json new_map_json")
    print("new_map_json is like: {\"1\":15}  (maps old id 1 -> cocoid 15)")
    sys.exit(1)

infile, outfile, mapfile = sys.argv[1], sys.argv[2], sys.argv[3]
j = json.load(open(infile))
mapping = json.load(open(mapfile))  # keys as strings, values ints; e.g. {"1":15}

# normalize mapping keys to ints
mapping = {int(k): int(v) for k, v in mapping.items()}

# Remap annotation category ids
for a in j.get("annotations", []):
    old = a.get("category_id")
    if old in mapping:
        a["category_id"] = mapping[old]

# Fix categories array: ensure each mapped id appears with name
# If categories empty or contain old ids, replace with mapped set.
new_cat_ids = sorted(set(mapping.values()))
# If categories are present, try to update names; otherwise, create fresh entries
cats = j.get("categories", [])
name_by_old = {c['id']: c.get('name', f"cat{c['id']}") for c in cats}
new_cats = []
for old, new in mapping.items():
    name = name_by_old.get(old, name_by_old.get(new, "bird"))
    new_cats.append({"id": new, "name": name})
j["categories"] = new_cats

# Fix file_name to basename (remove any folders/backslashes)
for im in j.get("images", []):
    im_fn = im.get("file_name", "")
    im["file_name"] = os.path.basename(im_fn)

for img in j.get("images", []):
    if "file_name" in img:
        clean_path = img["file_name"].replace("\\", "/")
        img["file_name"] = os.path.basename(clean_path)

json.dump(j, open(outfile, "w"), indent=2)
print("Wrote", outfile)
print("Mapping:", mapping)
