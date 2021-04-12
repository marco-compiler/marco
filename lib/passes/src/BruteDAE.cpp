#include "modelica/passes/BruteDAE.hpp"

using namespace modelica;
using namespace std;
using namespace llvm;

Expected<AssignModel> modelica::addJacobianAndResidual(Model& model)
{
	AssignModel out;

	return out;
}

// Calculate Jacobian Matrix from Model (here or in ModJacobian)

// Calculate derivatives for every equation (here or in ModJacobian)
// Calculate derivatives for every expression (here or in ModJacobian)

// Compute residual function for every equation (here or in ModResidual)
// Compute residual function for every expression (here or in ModResidual)

// Create all functions needed by SUNDIALS IDA in a suitable format
