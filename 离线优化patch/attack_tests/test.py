import pickle
METADATA_PATH   = '../models/weights/supercombo_metadata.pkl'
with open(METADATA_PATH, 'rb') as f:
    metadata = pickle.load(f)
a = metadata.get('output_slices', {}).keys()
b = metadata.get('output_shapes', {})
print("Output slices keys:", metadata.get('output_slices', {}).keys())
print("Output shapes:", metadata.get('output_shapes', {}))