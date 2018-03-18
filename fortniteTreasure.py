# Download chest locations
import urllib.request, json
DATA_ADDRESS = "http://www.fortnitechests.info/api/chests"
with urllib.request.urlopen(DATA_ADDRESS) as url:
    data = json.loads(url.read().decode())
data = data['lootchests']

# Scale the data to be between 0-1
import numpy as np
MAX_DIMENSION = 2000.0  # Raw data can be anywhere from 0-2000
longitude = np.zeros((len(data)))
latitude = np.zeros((len(data)))
X_raw = np.zeros((len(data), 2))
for i in range(longitude.shape[0]):
    X_raw[i, 0] = data[i]['lng']
    X_raw[i, 1] = data[i]['lat']

# Correct for the map curveature in the X direction
X_raw[:, 0] = X_raw[:, 0] - (0.0237148 * X_raw[:, 0] + 0.8775403)
X = X_raw / MAX_DIMENSION

# Plot the data and save it to file
import matplotlib.pyplot as plt
from pylab import savefig

fig = plt.figure(0)
ax = fig.gca()
ax.set_xticks(np.arange(0, 1, 0.1))
ax.set_yticks(np.arange(0, 1., 0.1))
plt.grid()
plt.scatter(X[:, 0], X[:, 1], color='yellow')
plt.xlim((0,1))
plt.ylim((0,1))
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Fortnite Loot Box Locations')
plt.savefig("rawdata.png")

# Use a gaussian mixture model to identify clusters of boxes
from sklearn.mixture import GaussianMixture
RANDOM_STATE = 66
N_CLUSTERS = 28
gmm = GaussianMixture(n_components=N_CLUSTERS, random_state=RANDOM_STATE)
gmm.fit(X)
predict = gmm.predict(X)
means = gmm.means_

# Plot the clusters centers that were found and overlay them
plt.scatter(means[:, 0], means[:, 1], color='blue')
plt.savefig("clusters.png")

# Remove any clusters that are not within a 30 second run of a cluster center
from scipy.spatial.distance import cdist
TIME_SECONDS = 30
UNITS_PER_SECOND = 4.34 # Calculated empirically
units_per_second_scaled = UNITS_PER_SECOND / MAX_DIMENSION
radius = units_per_second_scaled * TIME_SECONDS
center_points = np.zeros((N_CLUSTERS, 2))

is_within_radius = np.zeros((X.shape[0], N_CLUSTERS))
for i in range(N_CLUSTERS):
    distances = cdist(X, np.expand_dims(means[i, :],axis=0))
    is_within_radius[:, i] = np.squeeze(distances < radius, axis=1)
outliers = np.sum(is_within_radius, axis=1) == 0

# Plot the outliers and then remove them
plt.scatter(X[outliers, 0], X[outliers, 1], color='red')
plt.savefig("outliers.png")
X = X[np.logical_not(outliers), :]
plt.close('all')

# Use KMeans clustering to identify new clusters on the dataset
from sklearn.cluster import KMeans
model = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
clusters = model.fit_predict(X)

# Plot the data clusters by color
fig = plt.figure(1)
ax = fig.gca()
ax.set_xticks(np.arange(0, 1, 0.1))
ax.set_yticks(np.arange(0, 1., 0.1))
plt.grid()
plt.scatter(X[:, 0], X[:, 1], c=clusters)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title("Fortnite Loot Box Clusters")
plt.xlim((0,1))
plt.ylim((0,1))
savefig("kmeans_clusters.png")

# Define a function for calculating the distance between two points relative to a predefined path
def calculate_path_distance(data, path):
    distance = np.zeros(path.shape[0]-1)
    for i in range(1, path.shape[0]):
        distance[i-1] = cdist(np.expand_dims(data[path[i], :], axis=0), np.expand_dims(data[path[i-1], :], axis=0))
    total_distance = np.sum(distance)
    return total_distance, distance

# Import the travelling salesperson code
from tsp import two_opt

# Initialize lists to store data
cluster_data_full = []
path_data_full = []
distance_data_full = []

# Iterate through each cluster and calculate the shortest route
for i in range(0, N_CLUSTERS):
    cluster_data = X[clusters==i, :]
    distance_list = []
    data_list = []
    path_list = []
    total_distance_array = np.zeros(cluster_data.shape[0])

    # Iterate through each loot box in the cluster, intializing the shortest path problem
    # with a different starting loot box each iteration
    for ii in range(0, cluster_data.shape[0]):
        rotated_data = np.roll(cluster_data, -ii, axis=0)
        path = two_opt(rotated_data, 0.00001)
        total_distance, distance = calculate_path_distance(rotated_data, path)
        total_distance_array[ii] = total_distance
        path_list.append(path)
        data_list.append(rotated_data)
        distance_list.append(distance)

    # Identify which the initializations achieved the shortest path
    minimum_distance = np.argmin(total_distance_array)

    # Append the shortest path information for the cluster
    cluster_data_full.append(data_list[minimum_distance])
    path_data_full.append(path_list[minimum_distance])
    distance_data_full.append(distance_list[minimum_distance])

# Rank the clusters by which obtain the most boxes in 60 seconds
TIME_THRESHOLD = 60
boxes_looted_in_threshold = np.zeros((N_CLUSTERS))

for i in range(0, N_CLUSTERS):
    time = distance_data_full[i] * (1/units_per_second_scaled)
    cumulative_sum = np.cumsum(time)
    boxes_looted_in_threshold[i] = cumulative_sum[cumulative_sum<TIME_THRESHOLD].shape[0]

# Choose the TOP_N to report the results on
TOP_N = 10
sorted_boxes = np.flip(np.argsort(boxes_looted_in_threshold), axis=0)
sorted_boxes = sorted_boxes[0:TOP_N]

# Select the relevant data
cluster_data_full = [cluster_data_full[i] for i in sorted_boxes]
path_data_full = [path_data_full[i] for i in sorted_boxes]
distance_data_full = [distance_data_full[i] for i in sorted_boxes]

# Create a plot of boxes checked vs. Time spent in the regions
plt.figure(2)
for i in range(0, TOP_N):
    time = distance_data_full[i] * (1/units_per_second_scaled)
    cumulative_sum = np.cumsum(time)
    boxes_checked = np.arange(1, time.shape[0]+1)
    path = path_data_full[i]
    plt.plot(cumulative_sum, boxes_checked)

LOCATIONS = ['Tilted Towers', 'Factory', 'Retail Row', 'Haunted Hills', 'Pleasant Park', 'Snobby Shores', 'Salty Springs',  'Flush Factory', 'Lonely Lodge',  'Greasy Grove']
plt.title('Boxes Checked vs. Time (seconds)')
plt.xlabel('Time (seconds)')
plt.ylabel('Boxes Checked')
plt.legend(LOCATIONS)
savefig("boxescheckedVtime.png")


# Download the map file
from PIL import Image
import os

# Download world map and resize it to be based on our scale
MAP_ADDRESS = "http://www.fortnitechests.info/assets/images/web/5KMap.png"
MAP_FILENAME = "map.png"
if not os.path.isfile(MAP_FILENAME):
    urllib.request.urlretrieve(MAP_ADDRESS, MAP_FILENAME)

int_max_dimension = int(MAX_DIMENSION)
map = Image.open(MAP_FILENAME)
plt.close('all')
map = map.resize((int_max_dimension, int_max_dimension))


for i in range(0, TOP_N):
    plt.close('all')

    # Select the cluster and path data and rescale
    path = path_data_full[i]
    cluster_data_select = cluster_data_full[i]
    cluster_data_select = cluster_data_select * MAX_DIMENSION

    # Flip the y axis
    cluster_data_select[:, 0] = cluster_data_select[:, 0]
    cluster_data_select[:, 1] = (MAX_DIMENSION - cluster_data_select[:, 1])

    # Correct for curvature in the y directio
    cluster_data_select[:, 1]  = cluster_data_select[:, 1] - (0.02336636 * cluster_data_select[:, 1] - 46.31481)

    # Reorder the data
    reordered_data = cluster_data_select[path, :]

    # Plot the data
    MARKER_SIZE = 0.05
    LINE_SIZE = 0.15

    plt.figure(2 + i)
    ax = plt.gca()
    plt.imshow(map)

    plt.plot(reordered_data[:, 0], reordered_data[:, 1], lw=LINE_SIZE, c='blue')
    ax.scatter(reordered_data[:, 0], reordered_data[:, 1], c='yellow', s=MARKER_SIZE)
    ax.scatter(reordered_data[0, 0], reordered_data[0, 1], c='green', s=MARKER_SIZE)
    ax.scatter(reordered_data[-1, 0], reordered_data[-1, 1], c='red', s=MARKER_SIZE)
    plt.axis('off')
    plt.title(reordered_data.shape[0])

    filename = str(i) + '.png'
    savefig(filename, dpi=1200)











