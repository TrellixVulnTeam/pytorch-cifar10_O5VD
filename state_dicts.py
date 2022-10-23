import argparse
import os
import sys

import requests
import tarfile
from tqdm import tqdm


class StateDicts:
    url = "https://cloud.riscue.xyz/s/7ZDb7k3t5btDecm/download"

    def run(self, args):
        if args.backup:
            self.backup()
        if args.upload:
            self.upload()

        if args.download:
            self.download()
        if args.extract:
            self.extract()

        if args.remove:
            self.remove()

    def upload(self):
        print('Upload not implemented!')

    def download(self):
        r = requests.get(self.url, stream=True)

        total_size = int(r.headers.get('content-length', 0))
        block_size = 2 ** 20
        t = tqdm(total=total_size, unit='MiB', unit_scale=True)

        with open('state_dicts.tar.gz', 'wb') as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()

        if total_size != 0 and t.n != total_size:
            raise Exception('Error, something went wrong')
        print('Download successful.')

    def backup(self):
        with tarfile.open('state_dicts.tar.gz', mode='w:gz') as archive:
            archive.add('state_dicts', recursive=True)
        print('Backup successful!')

    def extract(self):
        path_to_tar_file = os.path.join(os.getcwd(), 'state_dicts.tar.gz')
        with tarfile.open(path_to_tar_file, 'r') as archive:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(archive, os.getcwd())
        print('Extract successful!')

    def remove(self):
        os.remove("state_dicts.tar.gz")
        print('Delete successful!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 State_Dicts downloader')
    parser.add_argument('-b', '--backup', action='store_true', help='Backup as state_dicts.tar.gz')
    parser.add_argument('-u', '--upload', action='store_true', help='Upload state_dicts.tar.gz')
    parser.add_argument('-d', '--download', action='store_true', help='Download state_dicts.tar.gz')
    parser.add_argument('-e', '--extract', action='store_true', help='Extract all files from state_dicts.tar.gz')
    parser.add_argument('-r', '--remove', action='store_true', help='Remove state_dicts.tar.gz')
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
    StateDicts().run(parser.parse_args())
