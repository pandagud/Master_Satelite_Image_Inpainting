# -*- coding: utf-8 -*-
import logging
import click
from pathlib import Path
from src.dataLayer.importRGB import importData

class mainDataLayer():

    def test(self):
        cur = importData()
        cur.Load()
        print("HALLO")