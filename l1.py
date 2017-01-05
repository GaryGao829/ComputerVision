#!/usr/bin/env python
# coding=utf-8
from PIL import Image
im = Image.open("/Users/yugao/Programming_computer_vision_with_python/pic.jpg").convert('L')
im.show()
