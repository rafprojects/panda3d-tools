from triposr import TripoSR

model = TripoSR.from_pretrained()
mesh = model.generate_3d_model("input_ring.jpg")
mesh.save("ring_model.obj")
