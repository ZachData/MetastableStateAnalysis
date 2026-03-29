#test.py


from transformers import AlbertModel
m = AlbertModel.from_pretrained('albert-base-v2')
layer = m.encoder.albert_layer_groups[0].albert_layers[0]
for name, mod in layer.named_children():
    print(f'{name}: {type(mod).__name__}')