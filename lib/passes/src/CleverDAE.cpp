#include "modelica/passes/CleverDAE.hpp"

using namespace modelica;
using namespace std;
using namespace llvm;

Expected<AssignModel> modelica::addBLTBlocks(Model& model)
{
	AssignModel out;

	return out;
}

// Calculate Jacobian Matrix for each Algebraic Loop from Model

// Compute Residual Function for every Algebraic Loop

// Create all functions needed by SUNDIALS IDA in a suitable format
