import findspark
import numpy as np
from haversine import haversine

findspark.init()
import pyspark

sc = pyspark.SparkContext()
path = "/Users/Rishi/Downloads/sample_geo.txt"


def get_data(path):
	data = sc.textFile(path)
	data = data.map(lambda line: parse_line(line))
	return data


def parse_line(line):
	args = [x for x in line.split(",")]
	point = (float(args[1]), float(args[0]))
	location = int(args[2])
	return [point, location]


def get_bounds(data):
	max_long = data.max(lambda row: row[0][0])[0][0]
	min_long = data.min(lambda row: row[0][0])[0][0]
	max_lat = data.max(lambda row: row[0][1])[0][1]
	min_lat = data.min(lambda row: row[0][1])[0][1]
	return max_long, min_long, max_lat, min_lat


def create_init_clusters(clusters, data):
	init_clusters = data.takeSample(False, clusters, 42)
	init_clusters = sc.parallelize([(1, row[0], (pow(row[0][0], 2), pow(row[0][1], 2))) for row in init_clusters])
	return init_clusters


def mahalanobis_distance(p, c, type="euclidean"):
	sum = 0
	if type == "euclidean":
		for i in range(2):
			sum += pow((p[i] - c[1][i] / c[0]) / (c[2][i] / c[0] - pow(c[1][i] / c[0], 2)), 2)
	else:
		haversine_dist = haversine(p[::-1],(p[1], c[1][0] / c[0]))
		sum += pow(haversine_dist / (c[2][0]/c[0] - pow(c[1][0] / c[0], 2)), 2)
		haversine_dist = haversine(p[::-1], (c[1][1] / c[0], p[0]))
		sum += pow(haversine_dist / (c[2][1] / c[0] - pow(c[1][1] / c[0], 2)), 2)
	return pow(sum, .5)





def show(x):
	print str(x)


# def closestPoint(p, points):
# 	closest = float('inf')
# 	best_point = None
# 	for i, point in enumerate(points):
# 		dist = euclidDistance(p, point)
# 		if dist < closest:
# 			closest = dist
# 			best_point = point
# 	return best_point
#
#
# def addPoints(p1, p2):
# 	return p1[0]+p2[0], p1[1]+p2[1]
#
#





data = get_data(path)
params = get_bounds(data)
print(params)