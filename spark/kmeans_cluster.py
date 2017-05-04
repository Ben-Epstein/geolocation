import sys
from itertools import islice
import findspark
from haversine import haversine

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
	init_clusters = sc.parallelize([[2, list(row), [pow(row[0], 2), pow(row[1], 2)], indx] for indx, row in enumerate(init_clusters)])
	return init_clusters

def assign_cluster(p, type="euclidean"):
	global clusters_loc
	min_dist = float("inf")
	assign_indx = None
	for i, c in enumerate(clusters_loc):
		cur_dist = mahalanobis_distance(p, c, type)
		if cur_dist < min_dist:
			min_dist = cur_dist
			assign_indx = i
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


if __name__ == "__main__":
	path = "/Users/Rishi/Downloads/sample_geo.txt"
	data = get_data(path)
	clusters = create_init_clusters(4)
	clusters_loc = clusters.collect()
	n = data.count()
	total = 0
	other = 0
	for i in range(0, n/2, 50):
		part = data.mapPartitions(lambda it: islice(it, i, i+50)).map(lambda p: assign_cluster(p)).keyBy(lambda arr: arr[-1])

		classifications = part.groupByKey().map(lambda x: reduce_points_to_cluster(list(x[1])))
		print(classifications.collect())
		other += classifications.count()
		total += part.count()

		# classifications = part.reduceByKey(lambda p1, p2 : ( ))

		# print part.collect()[0:10]
		# print "********"
	print total
	print other
	print n
