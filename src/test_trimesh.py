import trimesh as tm
import numpy as np

vertices = np.array([[0, 0, 0], [1, 1, 1], [-1, 1, 1]])
lines = [tm.path.entities.Line([0, 1]),
         tm.path.entities.Line([1, 2]),
         tm.path.entities.Line([2, 0])]
# per entity RGB or RGBA colors, alpha is ignored in "notebook" viewer
colors = [(255, 0, 0, 255),(0, 255, 0, 127),(0, 0, 255, 32)]
p = tm.path.Path3D(entities=lines, vertices=vertices, process=False, colors=colors)
sc = tm.Scene(p)
sc.show()