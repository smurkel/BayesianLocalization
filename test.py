# How does one start an analysis?

data = Load("tiffstack.tif")

model = Model(data)

model.add_particle()
model.optimize_particle()

