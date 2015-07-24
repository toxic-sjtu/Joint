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
	std::vector<std::vector<struct model*>> models;
	Shape meanShape;
	std::vector<Sample> samples;
	int totalStages;
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
		const vector<Sample>& samples
		);
	void GlobalRegression(
		feature_node ** binfeatures,
		const std::vector<Sample> &samples,
		int stages
		);

	void Train(const std::vector<Sample>& samples);
};

#endif