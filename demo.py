import visdom
import numpy as np
cfg = {"server": "localhost",
       "port": 8097}
vis = visdom.Visdom('http://' + cfg["server"], port=cfg["port"])
vis.text('Hello Visdom!')
image = np.random.randint(0, 255, (3, 256, 256), dtype=np.uint8)
vis.image(image)

print(' gegeg')
vis.line(
       X=np.stack([np.array([1.5])] * 5, 1),
       Y=[np.array([20, 51.3, 95.0, 51.5, 24.11])],
       opts=dict(
              title='loss over time',
              legend=['A', 'B', 'C', 'D', 'E'],
              xlabel='epoch',
              ylabel='loss',
       ),
       win=1,
)
