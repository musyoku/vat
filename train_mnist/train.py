import numpy as np
import os, sys, time
from chainer import cuda
from chainer import functions as F
import pandas as pd
sys.path.append(os.path.split(os.getcwd())[0])
import dataset
from progress import Progress
from model import vat
from args import args

def main():
	# load MNIST images
	images, labels = dataset.load_train_images()

	# config
	config = vat.config

	# settings
	max_epoch = 1000
	num_trains_per_epoch = 500
	batchsize_l = 100
	batchsize_u = 200

	# seed
	np.random.seed(args.seed)
	if args.gpu_device != -1:
		cuda.cupy.random.seed(args.seed)

	# save validation accuracy per epoch
	csv_results = []

	# create semi-supervised split
	num_validation_data = 10000
	num_labeled_data = 100
	training_images_l, training_labels_l, training_images_u, validation_images, validation_labels = dataset.create_semisupervised(images, labels, num_validation_data, num_labeled_data, config.ndim_y, seed=args.seed)
	print training_labels_l

	# training
	progress = Progress()
	for epoch in xrange(1, max_epoch):
		progress.start_epoch(epoch, max_epoch)
		sum_loss_supervised = 0
		sum_loss_lds = 0

		for t in xrange(num_trains_per_epoch):
			# sample from data distribution
			images_l, label_onehot_l, label_ids_l = dataset.sample_labeled_data(training_images_l, training_labels_l, batchsize_l, config.ndim_x, config.ndim_y, binarize=False)
			images_u = dataset.sample_unlabeled_data(training_images_u, batchsize_u, config.ndim_x, binarize=False)

			# supervised loss
			unnormalized_y_distribution = vat.encode_x_y(images_l, apply_softmax=False)
			loss_supervised = F.softmax_cross_entropy(unnormalized_y_distribution, vat.to_variable(label_ids_l))

			# virtual adversarial training
			lds_l = -F.sum(vat.compute_lds(images_l)) / batchsize_l
			lds_u = -F.sum(vat.compute_lds(images_u)) / batchsize_u
			loss_lsd = lds_l + lds_u

			# backprop
			vat.backprop(loss_supervised + config.lamda * loss_lsd)

			sum_loss_supervised += float(loss_supervised.data)
			sum_loss_lds += float(loss_lsd.data)
			if t % 10 == 0:
				progress.show(t, num_trains_per_epoch, {})

		vat.save(args.model_dir)

		# validation
		images_l, _, label_ids_l = dataset.sample_labeled_data(validation_images, validation_labels, num_validation_data, config.ndim_x, config.ndim_y, binarize=False)
		images_l_segments = np.split(images_l, num_validation_data // 500)
		label_ids_l_segments = np.split(label_ids_l, num_validation_data // 500)
		sum_accuracy = 0
		for images_l, label_ids_l in zip(images_l_segments, label_ids_l_segments):
			y_distribution = vat.encode_x_y(images_l, apply_softmax=True, test=True)
			accuracy = F.accuracy(y_distribution, vat.to_variable(label_ids_l))
			sum_accuracy += float(accuracy.data)
		validation_accuracy = sum_accuracy / len(images_l_segments)
		
		progress.show(num_trains_per_epoch, num_trains_per_epoch, {
			"loss_spv": sum_loss_supervised / num_trains_per_epoch,
			"loss_lds": sum_loss_lds / num_trains_per_epoch,
			"accuracy": validation_accuracy,
		})

		# write accuracy to csv
		csv_results.append([epoch, validation_accuracy, progress.get_total_time()])
		data = pd.DataFrame(csv_results)
		data.columns = ["epoch", "accuracy", "min"]
		data.to_csv("{}/result.csv".format(args.model_dir))

if __name__ == "__main__":
	main()
