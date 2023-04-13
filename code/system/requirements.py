import os
import sys

print("Python version")
print(sys.version)
print("Version info.")
print(sys.version_info)
os.system("pip freeze > requirements.txt")