// 
//  jointRegressor.h
//   
//  Toxic
//

#ifndef JOINT_REGRESSOR_H
#define JOINT_REGRESSOR_H

#include "randomForest.h"
#include "linear.h"

class JointRegressor {
public:
	std::vector<RandomForest> forests;
	std::vector<std::vector<model*>> models;
	Shape meanShape;
	std::vector<Sample> posSamples;
	std::vector<Sample> samples;
	int totalStages;
	feature_node ** binFeatures;
	feature_node ** posBinFeatures;
public:
	JointRegressor() {
		totalStages = GlobalParams::stages;
		forests.resize(totalStages);
		models.resize(totalStages);
	}
	~JointRegressor() {

	}
	
	feature_node ** DeriveBinaryFeat(
		const RandomForest& forest,
		const std::vector<Sample>& samples
		);
	void ReleaseFeatureSpace(
		feature_node ** binfeatures,
		int numSamples);
	int GetCodeFromTree(
		const Tree& tree,
		const Sample& sample,
		const Mat_<double> &rotation,
		const double scale
		);
	void GlobalRegression(int stages);

	void Train(std::vector<Sample>& samples);
	void ReleaseFeatures(feature_node ** binFeatures, int numSamples);
};

#endif