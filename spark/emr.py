import sys
from math import radians, cos, sin, asin, sqrt
import findspark
import numpy.random as rnd
from itertools import islice
findspark.init()
import pyspark
sc = pyspark.SparkContext()
#Code contained in Haversine module, which could not be imported in EMR
def haversine(point1, point2, miles=False):
        AVG_EARTH_RADIUS = 6371
    # unpack latitude/longitude
        lat1, lng1 = point1
        lat2, lng2 = point2
    # convert all latitudes/longitudes from decimal degrees to radians
        lat1, lng1, lat2, lng2 = map(radians, (lat1, lng1, lat2, lng2))
    # calculate haversine
        lat = lat2 - lat1
        lng = lng2 - lng1
        d = sin(lat * 0.5) ** 2 + cos(lat1) * cos(lat2) * sin(lng * 0.5) ** 2
        h = 2 * AVG_EARTH_RADIUS * asin(sqrt(d))
        if miles:
            return h * 0.621371  # in miles
        else:
            return h
#Load data into the SparkContext, in this case persistently
def get_data(path):
	data = sc.textFile(path)
	data = data.map(lambda line: parse_line(line))
	data.persist(pyspark.StorageLevel.MEMORY_AND_DISK)
	return data

#Extract the latitude, longitude values from the line of text from data
def parse_line(line):
	args = [x for x in line.split(",")]
	point = (float(args[1]), float(args[0]))
	return [point, 1]

#Create the first clusters, by taking a pseudorandom sample of the data for
#a given k
def create_init_clusters(data, k):
	sample = data.takeSample(False, 2*k, 42)
	init_clusters = []
    #increment in steps of two for the assignment to clusters
	for i, j in zip(range(0, 2*k + 1, 2), range(2, 2*k + 1, 2)):
		val1 = sample[i]
		val2 = sample[j-1]
        #compute the sum and squares of values in the clusters
        #use values to compute variance for the width and height of ellipses to
        #define clusters
		init_clusters.append((val1[0][0] + val2[0][0], val1[0][1] + val2[0][1]))
	init_clusters = sc.parallelize([[2,
									 list(row),
									 [pow(row[0], 2), pow(row[1], 2)],
									 indx] for indx, row in enumerate(init_clusters)]).keyBy(lambda arr: arr[-1])
        init_clusters.cache()
	return init_clusters
#find points with the minimum distance to a given cluster to assign to that cluster
#default to Euclidean distance
def assign_cluster(p, type="euclidean"):
	global clusters_loc
	min_dist = float("inf")
	assign_indx = None
	for c in clusters_loc:
		cur_dist = mahalanobis_distance(p, c[1], type)
		if cur_dist < min_dist:
			min_dist = cur_dist
			assign_indx = c[1][-1]
	return p + [assign_indx]
#distance function, either Haversine or Euclidean
def mahalanobis_distance(p, c, type="euclidean"):
	sum = 0
	if type == "euclidean":
		for i in range(2):
			numerator = pow(p[0][i] - c[1][i]/c[0], 2)
			denominator = (c[2][i] / c[0]) - pow(c[1][i]/c[0], 2)
			sum += numerator/denominator
	else:
		#calculate the Haversine or Great Circle distance based on haversine function
		haversine_dist = haversine(p[::-1][1], (float(p[1]), float(c[1][0] / c[0])))
		sum += pow(haversine_dist / (c[2][0]/c[0] - pow(c[1][0]/c[0], 2)), 2)
	return pow(sum, .5)

#assign new values to a cluster based on aggreagted points
def reduce_points_to_cluster(points_list):
	cluster = [0, [0, 0], [0, 0], points_list[0][-1]]
	for p in points_list:
		cluster[0] += 1
		cluster[1][0] += p[0][0]
		cluster[1][1] += p[0][1]
		cluster[2][0] += pow(p[0][0], 2)
		cluster[2][1] += pow(p[0][1], 2)
	return cluster

#format new clusters
def add_clusters(c1, c2):
	cluster = [0, [0, 0], [0, 0], c1[-1]]
	cluster[0] = c1[0] + c2[0]
	cluster[1][0] = c1[1][0] + c2[1][0]
	cluster[1][1] = c1[1][1] + c2[1][1]
	cluster[2][0] = c1[2][0] + c2[2][0]
	cluster[2][1] = c1[2][1] + c2[2][1]
	return cluster
#find if threshold has been reached
def converge_dist_check(old_clusters, new_clusters, threshold=.1):
	for old, new in zip(old_clusters, new_clusters):
		old_center, new_center = (old[1][1][0]/old[1][0], old[1][1][1]/old[1][0]), (new[1][1][0]/new[1][0], new[1][1][1]/new[1][0])
		if abs(old_center[0] - new_center[0]) < threshold and abs(old_center[1] - new_center[1]) < threshold:
			continue
		else:
			return False
	return True

#Main function, executes job
if __name__ == "__main__":
	data = get_data(sys.argv[1])
    #current number of clusters
	k = 4
    #length of the dataset
	n = data.count()
	clusters = create_init_clusters(data, k)
	clusters_loc = clusters.collect()
    #iterate from 0 to datasize/2, incremented in steps of 50
	for i in range(0, n/2, 50):
        #create a partition for the dataset, assign its rows to clusters, and create a key for each based on its assigned cluster
		part = data.mapPartitions(lambda it: islice(it, i, i+10)).map(lambda p: assign_cluster(p)).keyBy(lambda arr: arr[-1])
        #create lists of values associated with a given key
		classifications = part.groupByKey().map(lambda x: reduce_points_to_cluster(list(x[1]))).keyBy(lambda arr: arr[-1])
        #reassign values of clusters based on points added to them
		clusters = sc.union([clusters, classifications]).reduceByKey(lambda v1, v2: add_clusters(v1, v2))
        #stop changing values of clusters when the threshold of 0.1 is reached
		if converge_dist_check(clusters_loc, clusters.collect()):
			break
		clusters_loc = clusters.collect()
