//
//	jointRegressor.cpp
//
//	implementation for joint classification and regression procedure
//
//	Toxic
//

#include "jointRegressor.h"

int JointRegressor::GetCodeFromTree(
	const Tree& tree,
	const Sample& sample,
	const Mat_<double>& rotation,
	const double scale
	) {
	int currentNode = 0;
	int binCode = 1;
	while (!tree.nodes[currentNode].isLeaf) {
		double x1 = tree.nodes[currentNode].feat[0].x;
		double y1 = tree.nodes[currentNode].feat[0].y;
		double x2 = tree.nodes[currentNode].feat[1].x;
		double y2 = tree.nodes[currentNode].feat[1].y;

		double project_x1 = rotation(0, 0) * x1 + rotation(0, 1) * y1;
		double project_y1 = rotation(1, 0) * x1 + rotation(1, 1) * y1;
		project_x1 = scale * project_x1 * sample.bb.width / 2.0;
		project_y1 = scale * project_y1 * sample.bb.height / 2.0;
		int real_x1 = project_x1 + sample.current(tree.landmarkID, 0);
		int real_y1 = project_y1 + sample.current(tree.landmarkID, 1);
		real_x1 = max(0.0, min((double)real_x1, sample.image.cols - 1.0));
		real_y1 = max(0.0, min((double)real_y1, sample.image.rows - 1.0));

		double project_x2 = rotation(0, 0) * x2 + rotation(0, 1) * y2;
		double project_y2 = rotation(1, 0) * x2 + rotation(1, 1) * y2;
		project_x2 = scale * project_x2 * sample.bb.width / 2.0;
		project_y2 = scale * project_y2 * sample.bb.height / 2.0;
		int real_x2 = project_x2 + sample.current(tree.landmarkID, 0);
		int real_y2 = project_y2 + sample.current(tree.landmarkID, 1);
		real_x2 = max(0.0, min((double)real_x2, sample.image.cols - 1.0));
		real_y2 = max(0.0, min((double)real_y2, sample.image.rows - 1.0));

		double difference = ((int)(sample.image(real_y1, real_x1)) -
			(int)(sample.image(real_y2, real_x2)));

		if (difference < tree.nodes[currentNode].threshold) {
			// go to its left child
			currentNode = tree.nodes[currentNode].cNodesID[0];
		}
		else {
			// go to its right child
			currentNode = tree.nodes[currentNode].cNodesID[1];
		}
	}
	for (int i = 0; tree.leafID.size(); i++) {
		if (tree.leafID[i] == currentNode) {
			break;
		}
		binCode++;
	}
	return binCode;
}

feature_node ** JointRegressor::DeriveBinaryFeat(
	const RandomForest& randomForest,
	const vector<Sample>& samples
	) {
	// allocate memory for binary features
	feature_node ** binFeatures;
	binFeatures = new feature_node* [samples.size()];
	for (int i = 0; i < samples.size(); i++) {
		binFeatures[i] =
			new feature_node[randomForest.numTrees * randomForest.numLandmarks + 1];
	}
	int binCode;
	int ind;
	int numNodes = pow(2, (randomForest.maxDepth - 1));
	Mat_<double> rotation;
	double scale;

	//extract feature for each sample
	for (int i = 0; i < samples.size(); i++) {
		Joint::SimilarityTransform(
			Joint::Project(samples[i].current, samples[i].bb),
			meanShape,
			rotation,
			scale);
		for (int j = 0; j < randomForest.numTrees; j++) {
			for (int k = 0; k < randomForest.numLandmarks; k++) {
				binCode = GetCodeFromTree(randomForest.trees[j][k],
					samples[i],
					rotation,
					scale);
				ind = k * randomForest.numTrees + j;
				binFeatures[i][ind].index = numNodes * ind + binCode;
				binFeatures[i][ind].value = 1;
			}
			binFeatures[i][randomForest.numLandmarks * randomForest.numTrees].index = -1;
			binFeatures[i][randomForest.numLandmarks * randomForest.numTrees].value = -1;
			
		}
	}
	return binFeatures;
}

void JointRegressor::GlobalRegression(int stages) {
	problem* prob = new problem;
	prob->l = posSamples.size();
	prob->n = GlobalParams::n_landmark 
		* GlobalParams::numTrees 
		* pow(2, (GlobalParams::depth - 1));
	prob->x = posBinFeatures;
	prob->bias = -1;

	parameter* param = new parameter;
	param->solver_type = L2R_L2LOSS_SVR_DUAL;
	param->C = 1.0 / posSamples.size();
	param->p = 0;

	int num_residual = GlobalParams::n_landmark * 2;

	double **yy = new double*[num_residual];
	for (int i = 0; i < num_residual; i++) {
		yy[i] = new double[posSamples.size()];
	}

	for (int i = 0; i < posSamples.size(); i++) {
		for (int j = 0; j < num_residual; j++) {
			Shape residual = Joint::GetShapeResidual(posSamples[i], meanShape);
			if (j < num_residual / 2) {
				yy[j][i] = residual(j, 0);
			}
			else{
				yy[j][i] = residual(j, 1);
			}
		}
	}

	models.clear();
	models.resize(num_residual);
	for (int i = 0; i < num_residual; i++) {
		prob->y = yy[i];
		check_parameter(prob, param);
		model* lbfModel = train(prob, param);
		models[stages][i] = lbfModel;
	}
	double tmp;
	double scale;
	Mat_<double> rotation;
	Mat_<double> deltaShape_bar(num_residual / 2, 2);
	Mat_<double> deltaShape_bar_trans(num_residual / 2, 2);
	for (int i = 0; i < samples.size(); i++) {
		for (int j = 0; j < num_residual; j++) {
			tmp = predict(models[stages][j], binFeatures[i]);
			if (j < num_residual / 2) {
				deltaShape_bar(j, 0) = tmp;
			}
			else {
				deltaShape_bar(j - num_residual / 2, 1) = tmp;
			}
		}

		Joint::SimilarityTransform(Joint::Project(samples[i].current, samples[i].bb),
			meanShape,
			rotation,
			scale);
		transpose(rotation, rotation);
		deltaShape_bar_trans = scale * deltaShape_bar * rotation;
		samples[i].current =
			Joint::ReProject((Joint::Project(samples[i].current, samples[i].bb) + deltaShape_bar_trans), samples[i].bb);
	}
}

void JointRegressor::Train(vector<Sample>& samples_) {
	samples = samples_;
	
	meanShape = Joint::GetMeanShape(posSamples);

	for (int stage = 0; stage < GlobalParams::stages; stage++) {
		forests[stage].Train(samples, meanShape, stage);

		posSamples.clear();
		for (int i = 0; i < samples.size(); i++) {
			if (samples[i].label == 1){
				posSamples.push_back(samples[i]);
			}
		}

		binFeatures = DeriveBinaryFeat(forests[stage], samples);
		posBinFeatures = DeriveBinaryFeat(forests[stage], posSamples);
		GlobalRegression(stage);
		ReleaseFeatures(binFeatures, samples.size());
		ReleaseFeatures(posBinFeatures, posSamples.size());
	}
}

void JointRegressor::ReleaseFeatures(feature_node** binFeatures, int numSamples) {
	for (int i = 0; i < numSamples; i++) {
		delete [] binFeatures[i];
		binFeatures[i] = nullptr;
	}
	delete [] binFeatures;
	binFeatures = nullptr;
}