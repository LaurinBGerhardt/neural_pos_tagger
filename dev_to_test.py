#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python 3.8

with open("dev.tagged","r",encoding="utf-8") as r:
    with open("test.tagged","w",encoding="utf-8") as w:
        for line in r.readlines():
            line = line.strip().split("\t")
            w.write(line[0])
            w.write("\n")
