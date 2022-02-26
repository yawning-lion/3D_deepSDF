import meshio

mesh = meshio.read('/home/yawnlion/Desktop/PYproject/3D_deepSDF/data/data_set/sphere.msh')
faces = mesh.get_cells_type("triangle")