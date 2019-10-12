#include "modelica/simulatorGenerator/Generator.hpp"

using namespace modelica;
using namespace llvm;

using E = SimExp;
using B = BultinSimTypes;
using T = SimType;

template<typename T>
using C = modelica::SimConst<T>;

bool makeSimulation(Lowerer& sim)
{
	if (!sim.addVar("intVector", E(C<int>(1, 2, 3), T(B::INT, 3))))
		return false;
	if (!sim.addVar("intVectorConstant", E(C<int>(-1, -2, -3), T(B::INT, 3))))
		return false;

	if (!sim.addUpdate(
					"intVector",
					E(C<int>(3, 3, 3), T(B::INT, 3)) + E("intVector", B::INT, 3)))
		return false;

	return true;
}
