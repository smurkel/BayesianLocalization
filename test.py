# How does one start an analysis?

data = Load("tiffstack.tif")

model = Model(data)

new_particle = model.add_particle()
model.set_active_particle(new_particle)

model.optimize_particle()

