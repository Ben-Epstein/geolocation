import sys
from itertools import islice
import findspark
from haversine import haversine
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy.random as rnd

findspark.init()
import pyspark


sc = pyspark.SparkContext()

def get_data(path):
	data = sc.textFile(path)
	data = data.map(lambda line: parse_line(line))
	return data


def parse_line(line):
	args = [x for x in line.split(",")]
	point = (float(args[1]), float(args[0]))
	return [point, 1]


def get_bounds():
	global data
	max_long = data.max(lambda row: row[0][0])[0][0]
	min_long = data.min(lambda row: row[0][0])[0][0]
	max_lat = data.max(lambda row: row[0][1])[0][1]
	min_lat = data.min(lambda row: row[0][1])[0][1]
	return max_long, min_long, max_lat, min_lat


def create_init_clusters(k):
	global data
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

def update(i):
	global clusters_loc, ax
	ellipses = [Ellipse(xy=[row[1][1][0]/row[1][0], row[1][1][1]/row[1][0]],
						width=pow(row[1][2][0]/row[1][0], .5)/10,
						height=pow(row[1][2][1]/row[1][0], .5)/10,
						label="Count: " + str(row[1][0])
						) for row in clusters_loc]

	for e in ellipses:
		ax.add_artist(e)
		e.set_clip_box(ax.bbox)
		e.set_facecolor(rnd.rand(3))

	# data = {"label": [row[0] for row in clusters_loc],
	# 		"count": [row[1][0] for row in clusters_loc],
	# 		"longitude": [row[1][1][0]/row[1][0] for row in clusters_loc],
	# 		"latitude": [row[1][1][1]/row[1][0] for row in clusters_loc],
	# 		"width": [pow(row[1][2][0]/row[1][0], .5)/10 for row in clusters_loc],
	# 		"height": [pow(row[1][2][1]/row[1][0], .5)/10 for row in clusters_loc]
	# }
	# return data

if __name__ == "__main__":
	path = "/Users/Rishi/Downloads/sample_geo.txt"
	data = get_data(path)
	clusters = create_init_clusters(4)
	clusters_loc = clusters.collect()
	print str(clusters_loc)
	fig, ax = plt.subplots()
	fig.set_tight_layout(True)

	plt.show()

	for i in range(0, 500, 25):

		part = data.mapPartitions(lambda it: islice(it, i, i+10)).map(lambda p: assign_cluster(p)).keyBy(lambda arr: arr[-1])
		classifications = part.groupByKey().map(lambda x: reduce_points_to_cluster(list(x[1]))).keyBy(lambda arr: arr[-1])
		clusters = sc.union([clusters, classifications]).reduceByKey(lambda v1, v2: add_clusters(v1, v2))
		clusters_loc = clusters.collect()
		print str(clusters_loc)

		# if i % 50 == 0:
		# 	plt = rdd_to_mpl(clusters_loc)
		# 	plt.show()




