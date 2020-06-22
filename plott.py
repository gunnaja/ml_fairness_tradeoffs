from plotter import plot_results
from filehandler import read_result_from_file



def plot():
	r1 = read_result_from_file('svm_21-05-2020_02-46.txt')
	r2 = read_result_from_file('svm_dir_21-05-2020_06-05.txt')
	r2 = read_result_from_file('svm_reweighing_21-05-2020_15-36.txt')
	r2 = read_result_from_file('svm_optimpreproc_21-05-2020_18-11.txt')
	plot_results([r1,r2,r3,r4])

plot()