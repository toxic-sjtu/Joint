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
};

#endif