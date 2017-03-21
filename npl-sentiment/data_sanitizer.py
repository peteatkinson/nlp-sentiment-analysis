import pandas as pd
import re as regex
import ftfy as fx
import unicodedata as ud
import sys

class SimpleDataModule():
    def __init__(self, dir):
        self.dir = dir

    def read_data(self):
        columns = ['review']
        df = pd.read_csv(self.dir, skipinitialspace=True, usecols=columns)
        return df.review.values

    def parse(self, raw):
        parser = regex.compile('<.*?>')
        data = regex.sub(parser, '', raw)
        data = data.replace('\n', '')
        encoding = fx.guess_bytes(data)[1]

        if encoding == 'utf-8':
            decode = data.decode('utf-8')
            data = fx.fix_text(decode)

        return data

    def chunkify(self,lst,n):
        return [ lst[i::n] for i in xrange(n) ]

    def load_data(self):
        data = self.read_data()
        dump = []

        for row in data:
            raw = self.parse(row)
            dump.append(ud.normalize('NFKD', raw).encode('ascii', 'ignore'))

        return self.chunkify(dump, 2)


