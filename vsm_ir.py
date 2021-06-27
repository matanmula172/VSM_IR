import nltk
import numpy as np
import json
import lxml.html


def read_f():
    with open('./cfc-xml_corrected/cf74.xml', 'r') as f:
        doc = lxml.html.fromstring(f.read())
        print(doc)
        rest = doc.xpath('/root/RECORD')
        for elem in rest:
            print(elem)


read_f() #Test
