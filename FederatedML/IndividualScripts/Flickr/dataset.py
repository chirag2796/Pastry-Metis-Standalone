import numpy as np

# Load data
with open('FlickrEdges2.txt', 'r') as f:
    lines = f.readlines()

# Extract edge list, feature matrix, and labels from file
edges = [line.strip().split('\t')[:2] for line in lines]
features = np.array([line.strip().split('\t')[2:-1] for line in lines], dtype=np.float32)

a = [line.strip().split('\t')[-1] for line in lines]
print(a[:5])

labels = np.array([int(line.strip().split('\t')[-1]) for line in lines], dtype=np.int32)

# Save data to separate files
np.savetxt('flickr.edges', edges, fmt='%s')
np.savetxt('flickr.features', features)
np.savetxt('flickr.labels', labels, fmt='%s')
