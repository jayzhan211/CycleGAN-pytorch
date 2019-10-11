import visdom
import numpy as np
cfg = {"server": "localhost",
       "port": 8097}
vis = visdom.Visdom('http://' + cfg["server"], port=cfg["port"])
vis.text('Hello Visdom!')
image = np.random.randint(0, 255, (3, 256, 256), dtype=np.uint8)
vis.image(image)