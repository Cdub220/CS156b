from pathlib import Path
import random
from PIL import Image

files = list(Path("/resnick/groups/CS156b/from_central/data/train").rglob("*.jpg"))
imgs = random.sample(files, 3)

for i, p in enumerate(imgs, 1):
    img = Image.open(p)
    img.show()
    print(i, p)