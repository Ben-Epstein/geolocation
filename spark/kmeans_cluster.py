import sys
from itertools import islice
import findspark
from haversine import haversine

findspark.init()
import pyspark


sc = pyspark.SparkContext()

def get_data(path):
	data = sc.textFile(path, numPartitions=64)
	data = data.map(lambda line: parse_line(line), preservesPartitioning=True)
	return data


def parse_line(line):
	args = [x for x in line.split(",")]
	point = (float(args[1]), float(args[0]))
	location = int(args[2])
	return [point, location]


def get_bounds():
	global data
	max_long = data.max(lambda row: row[0][0])[0][0]
	min_long = data.min(lambda row: row[0][0])[0][0]
	max_lat = data.max(lambda row: row[0][1])[0][1]
	min_lat = data.min(lambda row: row[0][1])[0][1]
	return max_long, min_long, max_lat, min_lat


def create_init_clusters(k):
	global data
	init_clusters = data.takeSample(False, k, 42)
	init_clusters = sc.parallelize([(1, row[0], (pow(row[0][0], 2), pow(row[0][1], 2))) for row in init_clusters])
	return init_clusters

# def reduce_points_to_cluster():


def mahalanobis_distance(p, c, type="euclidean"):
	sum = 0
	p = p[0]
	if type == "euclidean":
		for i in range(2):
			sum += pow((p[i] - c[1][i] / c[0]) / (c[2][i] / c[0] - pow(c[1][i] / c[0], 2)), 2)
	else:
		haversine_dist = haversine(p[::-1], (p[1], c[1][0] / c[0]))
		sum += pow(haversine_dist / (c[2][0]/c[0] - pow(c[1][0]/c[0], 2)), 2)
		haversine_dist = haversine(p[::-1], (c[1][1] / c[0], p[0]))
		sum += pow(haversine_dist / (c[2][1]/c[0] - pow(c[1][1] / c[0], 2)), 2)
	return pow(sum, .5)


def assign_cluster(p, type="euclidean"):
	min_dist = float("inf")
	assign_indx = None
	for i, c in enumerate(clusters):
		cur_dist = mahalanobis_distance(p, c, type)
		if cur_dist < min_dist:
			min_dist = cur_dist
			assign_indx = i
	return p + [assign_indx]


def add_point_to_cluster(c, p):
	c[0] += 1
	c[1][0] += p[0][0]
	c[1][1] += p[0][1]
	c[2][0] += pow(p[0][0], 2)
	c[2][1] += pow(p[0][1], 2)
	return c


if __name__ == "__main__":
	path = "/Users/Rishi/Downloads/sample_geo.txt"
	data = get_data(path)
	clusters = create_init_clusters(4)
	for i in range(64):
		part = data.mapPartitions(lambda it: islice(it, i-1, i)).map(lambda p: assign_cluster(p))
		classifications = part.reduceByKey(lambda row: row[-1])

	sample = data.takeSample(False, 100, 42)


	data.map(lambda p: assign_cluster(p))
	print clusters

