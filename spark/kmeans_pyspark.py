import sys

import findspark
import json
import numpy.random as rnd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from itertools import islice
from haversine import haversine

findspark.init()
import pyspark


sc = pyspark.SparkContext()

def get_data(path):
	data = sc.textFile(path)
	data = data.map(lambda line: parse_line(line))
	data.persist(pyspark.StorageLevel.MEMORY_AND_DISK)
	return data


def parse_line(line):
	args = [x for x in line.split(",")]
	point = (float(args[1]), float(args[0]))
	return [point, 1]


def get_bounds(data):
	max_long = int(data.max(lambda row: row[0][0])[0][0]) + 1
	min_long = int(data.min(lambda row: row[0][0])[0][0]) - 1
	max_lat = int(data.max(lambda row: row[0][1])[0][1]) + 1
	min_lat = int(data.min(lambda row: row[0][1])[0][1]) - 1
	return max_long, min_long, max_lat, min_lat


def create_init_clusters(data, k):
	sample = data.takeSample(False, 2*k, 42)
	init_clusters = []
	for i, j in zip(range(0, 2*k + 1, 2), range(2, 2*k + 1, 2)):
		val1 = sample[i]
		val2 = sample[j-1]
		init_clusters.append((val1[0][0] + val2[0][0], val1[0][1] + val2[0][1]))
	init_clusters = sc.parallelize([[2,
									 list(row),
									 [pow(row[0], 2), pow(row[1], 2)],
									 indx] for indx, row in enumerate(init_clusters)]).keyBy(lambda arr: arr[-1])
	init_clusters.cache()
	return init_clusters

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

def mahalanobis_distance(p, c, type="euclidean"):
	sum = 0
	if type == "euclidean":
		for i in range(2):
			numerator = pow(p[0][i] - c[1][i]/c[0], 2)
			denominator = (c[2][i] / c[0]) - pow(c[1][i]/c[0], 2)
			sum += numerator/denominator
	else:
		# sorry lmao
		haversine_dist = haversine(p[::-1], (p[1], c[1][0] / c[0]))
		sum += pow(haversine_dist / (c[2][0]/c[0] - pow(c[1][0]/c[0], 2)), 2)
		haversine_dist = haversine(p[::-1], (c[1][1] / c[0], p[0]))
		sum += pow(haversine_dist / (c[2][1]/c[0] - pow(c[1][1] / c[0], 2)), 2)
	return pow(sum, .5)


def reduce_points_to_cluster(points_list):
	cluster = [0, [0, 0], [0, 0], points_list[0][-1]]
	for p in points_list:
		cluster[0] += 1
		cluster[1][0] += p[0][0]
		cluster[1][1] += p[0][1]
		cluster[2][0] += pow(p[0][0], 2)
		cluster[2][1] += pow(p[0][1], 2)
	return cluster


def add_clusters(c1, c2):
	cluster = [0, [0, 0], [0, 0], c1[-1]]
	cluster[0] = c1[0] + c2[0]
	cluster[1][0] = c1[1][0] + c2[1][0]
	cluster[1][1] = c1[1][1] + c2[1][1]
	cluster[2][0] = c1[2][0] + c2[2][0]
	cluster[2][1] = c1[2][1] + c2[2][1]
	return cluster


def clusters_to_dict(clusters_loc):
	data = {"label": [row[0] for row in clusters_loc],
			"count": [row[1][0] for row in clusters_loc],
			"longitude": [row[1][1][0] / row[1][0] for row in clusters_loc],
			"latitude": [row[1][1][1] / row[1][0] for row in clusters_loc],
			"width": [pow(row[1][2][0] / row[1][0], .5) / 10 for row in clusters_loc],
			"height": [pow(row[1][2][1] / row[1][0], .5) / 10 for row in clusters_loc]
			}
	return data


def converge_dist_check(old_clusters, new_clusters, threshold=.1):
	for old, new in zip(old_clusters, new_clusters):
		old_center, new_center = (old[1][1][0]/old[1][0], old[1][1][1]/old[1][0]), (new[1][1][0]/new[1][0], new[1][1][1]/new[1][0])
		if abs(old_center[0] - new_center[0]) < threshold and abs(old_center[1] - new_center[1]) < threshold:
			continue
		else:
			return False
	return True


if __name__ == "__main__":
	path = "/Users/Rishi/Downloads/sample_geo.txt"
	data = get_data(path)
	bounds = get_bounds(data)
	k = 4
	iter_count = 0
	n = data.count()
	clusters = create_init_clusters(data, k)
	clusters_loc = clusters.collect()
	clusters_hist = {"{0}".format(iter_count): clusters_to_dict(clusters_loc),
					 "max_long":bounds[0],
					 "min_long": bounds[1],
					 "max_lat": bounds[2],
					 "min_lat": bounds[3]}

	for i in range(0, n/2, 50):
		part = data.mapPartitions(lambda it: islice(it, i, i+10)).map(lambda p: assign_cluster(p)).keyBy(lambda arr: arr[-1])
		classifications = part.groupByKey().map(lambda x: reduce_points_to_cluster(list(x[1]))).keyBy(lambda arr: arr[-1])
		clusters = sc.union([clusters, classifications]).reduceByKey(lambda v1, v2: add_clusters(v1, v2))
		if converge_dist_check(clusters_loc, clusters.collect()):
			break
		clusters_loc = clusters.collect()
		iter_count += 1
		clusters_hist["{0}".format(iter_count)] = clusters_to_dict(clusters_loc)

		# if iter_count % 50:
	with open("output/sample_clusters.json", "a") as f:
		json.dump(clusters_hist, f, indent=2)
	f.close()
