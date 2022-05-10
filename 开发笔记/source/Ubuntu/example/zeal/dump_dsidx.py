#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re, sqlite3
from bs4 import BeautifulSoup, NavigableString, Tag 

docpath = '/home/helios/application/TensorRT-8.2.2.1/doc/tensorrt.docset/Contents/Resources/Documents'
conn = sqlite3.connect("/home/helios/application/TensorRT-8.2.2.1/doc/tensorrt.docset/Contents/Resources/docSet.dsidx")
cur = conn.cursor()

try: cur.execute('DROP TABLE searchIndex;')
except: pass
cur.execute('CREATE TABLE searchIndex(id INTEGER PRIMARY KEY, name TEXT, type TEXT, path TEXT);')
cur.execute('CREATE UNIQUE INDEX anchor ON searchIndex (name, type, path);')

page = open(os.path.join(docpath,'index.html')).read()
soup = BeautifulSoup(page)

any = re.compile('.*')
for tag in soup.find_all('a', {'href':any}):
    name = tag.text.strip()
    if len(name) > 1:
        path = tag.attrs['href'].strip()
        if path != 'index.html':
            cur.execute('INSERT OR IGNORE INTO searchIndex(name, type, path) VALUES (?,?,?)', (name, 'func', path))
            print('name: %s, path: %s' % (name, path))

conn.commit()
conn.close()
print("ok")

