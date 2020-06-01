import tarfile

with tarfile.open('state_dicts.tar.gz', mode='w:gz') as archive:
    archive.add('./state_dicts', recursive=True)
