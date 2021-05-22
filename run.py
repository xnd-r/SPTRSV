from subprocess import Popen, PIPE
import datetime
import sys
import os
from glob import glob


def single_launch(name):
	proc = Popen(name, shell=True, stdout=PIPE, stderr=PIPE)
	proc.wait()
	res = proc.communicate()
	log = res[1].decode("utf-8")
	try:
		time = float(log.split('\n')[-2].split("Time: ")[-1])
		return log, time
	except ValueError as ve:
		print(log)
		with open(os.path.join(os.getcwd(), 'error_log.log'), 'w') as f:
			f.write(log)
		print("Runtime error. Exit")
		exit(0)

def min_res(name, repeats=3):
	min_time = 1e6
	min_log = ""
	for _ in range(repeats):
		log, time = single_launch(name)
		if time < min_time:
			min_time = time
			min_log = log
	return min_log, min_time

def get_times(filename, algos, matrices, repeats):
	run_name = datetime.datetime.now().strftime('%m_%d_%H_%M_%S')
	os.makedirs(os.path.join("experiments", run_name))
	global_log = ""
	prefix = "srun -N 1 -p gpu -t 720 "
	with open(os.path.join("experiments", run_name, "times.csv"), "w") as f:
		f.write(";")
		for a in algos:
			f.write(a + ';')
		f.write("\n")
		for m in matrices:
			print(m)
			f.write(os.path.basename(m) + ";")
			snodes = m[:-6] + "snodes"
			for a in algos:
				string = prefix + os.path.join(os.getcwd(), filename + " " + a + " " + m + " " + snodes + " " + "1 100 opt_true")
				print(string)
				log, time = min_res(string, repeats=repeats)
				global_log += log
				global_log += "-"*100+"\n"
				f.write(str(time) + ";")
			f.write("\n")
	with open(os.path.join("experiments", run_name, "best_times_log.log"), "w") as f:
		f.write(global_log)

def main():
	filename = os.path.join("code", "SPTRSV")
	algos = ["base", "custom"]
	matrices = glob("./matrices/bin/*")
	assert len(matrices) % 2 == 0
	matrices = list(filter(lambda x: not x.endswith('.snodes'), matrices))
	repeats = 4

	get_times(filename, algos, matrices, repeats)


if __name__ == "__main__":
	main()
