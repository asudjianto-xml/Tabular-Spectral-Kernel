"""Download public benchmark inputs into an explicit cache directory."""
from pathlib import Path
import hashlib,io,json,time,urllib.request,zipfile
import numpy as np
import pandas as pd

SOURCES={
 'adult':'https://archive.ics.uci.edu/static/public/2/adult.zip',
 'bike':'https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip',
 'taiwan':'https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip',
}
TEXT_HASHES={
 'adult.data':'5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d',
 'adult.test':'a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05',
 'hour.csv':'e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f',
}

def _download(url,path):
    if path.exists():return
    error=None
    for attempt in range(3):
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'Tabular-Spectral-Kernel/0.1 (research reproducibility)'})
            with urllib.request.urlopen(request,timeout=120) as r:content=r.read()
            temporary=path.with_suffix(path.suffix+'.partial');temporary.write_bytes(content);temporary.replace(path);return
        except (OSError,TimeoutError) as exc:error=exc;time.sleep(attempt+1)
    raise RuntimeError(f'Cannot download {url}. Check network access or place the original archive at {path}.') from error

def _member(archive,suffix):
    matches=[n for n in archive.namelist() if n.lower().endswith(suffix.lower())]
    if len(matches)!=1:raise ValueError(f'Expected one {suffix} in archive; found {matches}')
    return archive.read(matches[0])

def prepare(dataset,data_dir):
    data_dir=Path(data_dir).expanduser().resolve();data_dir.mkdir(parents=True,exist_ok=True)
    if dataset=='california':
        from sklearn.datasets import fetch_california_housing
        fetch_california_housing(data_home=data_dir/'california',download_if_missing=True)
        return data_dir
    if dataset not in SOURCES:raise ValueError(dataset)
    names={'adult':['adult.data','adult.test'],'bike':['hour.csv'],'taiwan':['taiwan_credit.npz']}[dataset]
    if not all((data_dir/name).exists() for name in names):
        archive=data_dir/f'{dataset}.zip';_download(SOURCES[dataset],archive)
        with zipfile.ZipFile(archive) as z:
            if dataset=='adult':
                for name in ['adult.data','adult.test','adult.names']:(data_dir/name).write_bytes(_member(z,name))
            elif dataset=='bike':
                if any(n.endswith('hour.csv') for n in z.namelist()):content=_member(z,'hour.csv')
                else:
                    with zipfile.ZipFile(io.BytesIO(_member(z,'.zip'))) as nested:content=_member(nested,'hour.csv')
                (data_dir/'hour.csv').write_bytes(content)
            else:
                df=pd.read_excel(io.BytesIO(_member(z,'.xls')),header=1,engine='xlrd')
                if df.shape!=(30000,25):raise ValueError(f'Unexpected Taiwan workbook shape {df.shape}')
                X=df.iloc[:,1:-1].to_numpy(np.float32);y=df.iloc[:,-1].to_numpy(np.int64)
                np.savez_compressed(data_dir/'taiwan_credit.npz',X=X,y=y)
    for name in names:
        if name in TEXT_HASHES:
            actual=hashlib.sha256((data_dir/name).read_bytes()).hexdigest()
            if actual!=TEXT_HASHES[name]:raise ValueError(f'{name} differs from the paper input; remove it and download again into a fresh cache.')
    if dataset=='taiwan':
        with np.load(data_dir/'taiwan_credit.npz') as z:
            expected=json.loads((Path(__file__).with_name('taiwan_signature.json')).read_text())
            for key in ['X','y']:
                actual=hashlib.sha256(z[key].tobytes()).hexdigest()
                if actual!=expected[key]:raise ValueError(f'Taiwan {key} array differs from the paper input')
    manifest={name:hashlib.sha256((data_dir/name).read_bytes()).hexdigest() for name in names}
    (data_dir/f'{dataset}_manifest.json').write_text(json.dumps(dict(source=SOURCES[dataset],files=manifest),indent=2)+'\n')
    return data_dir
