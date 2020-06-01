import requests, zipfile, os
from tqdm import tqdm

url = "https://rutgers.box.com/shared/static/y9wi8ic7bshe2nn63prj9vsea7wibd4x.zip"

r = requests.get(url, stream=True)

total_size = int(r.headers.get('content-length', 0))
block_size = 2**20
t=tqdm(total=total_size, unit='MiB', unit_scale=True)

with open('state_dicts.zip', 'wb') as f:
    for data in r.iter_content(block_size):
        t.update(len(data))
        f.write(data)
t.close()

if total_size != 0 and t.n != total_size:
    raise Exception('Error, something went wrong')

print('Download successful. Unzipping file.')
path_to_zip_file = os.path.join(os.getcwd(), 'state_dicts.zip')
with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall(os.getcwd())
    print('Unzip file successful!')

os.remove("state_dicts.zip")
