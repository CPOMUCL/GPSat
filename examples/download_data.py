import gdown
import zipfile
import os

from PyOptimalInterpolation import get_parent_path

# TODO: tidy this up, wrap into a function/method
print("Downloading data")
data_dir = get_parent_path("data")

# list of google drive file ids and out file names
id_file = [
    {"id": "1djlaZ2EKbm9pNAEt3w58WJtBA4NyQsNE", "file": "new_aux.zip", "unzip": True, "result": "aux"},
    {"id": "1cIh9lskzmL6C7EYV8lmJJ5YaJgKqOZHT", "file": "CS2S3_CPOM.zip", "unzip": True, "result": "CS2S3_CPOM"},
    {"id": "1gXsvtxZcWpBALomgeqn9kcfyCtKD3fkz", "file": "raw_along_track.zip", "unzip": True, "result": "RAW"},
    {"id": "1V1YbmSu10GzfQVCI75RRJdTozkwaUu1S", "file": "RAW/gpod_202003.h5", "unzip": False,
     "result": "RAW/gpod_202003.h5"}
    # this file is too big to download ?
    # {"id": "1rfNwiQ80LALDdJsYolfr-lVLRjRsjyVG", "file": "RAW/gpod_202002_202004_single.zip", "unzip": True,
    #  "result": "RAW/gpod_202002_202004_single.h5"},

]

for _ in id_file:
    id = _['id']
    file = _['file']
    unzip = _.get("unzip", False)
    result = _.get('result', "")

    # skip downloading if 'result' exists already
    if os.path.exists(os.path.join(data_dir, result)):
        print("-" * 10)
        print(f"skipping id: {id} as {os.path.join(data_dir, result)} already exists")
        continue

    # put data in data dir in repository
    output = os.path.join(data_dir, file)
    # make dir if need by
    os.makedirs(os.path.dirname(output), exist_ok=True)
    gdown.download(id=id, output=output, use_cookies=False)

    # unzip and remove zip file
    if unzip:
        print("unzipping")
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(path=data_dir)

        # remove zip folder
        os.remove(os.path.join(data_dir, file))

