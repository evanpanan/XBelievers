import sys
import os

# 确保根目录在 path 中
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from server import app
