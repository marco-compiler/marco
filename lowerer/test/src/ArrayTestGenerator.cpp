#include "modelica/simulatorGenerator/Generator.hpp"

using namespace modelica;
using namespace llvm;

using E = SimExp;
using B = BultinSimTypes;
using T = SimType;

template<typename T>
using C = modelica::SimConst<T>;

constexpr size_t three = 3;

bool makeSimulation(Simulation& sim)
{
	if (!sim.addVar("intVector", E(C<int>(1, 2, 3), T(B::INT, three))))
		return false;
	if (!sim.addVar("intVectorConstant", E(C<int>(-1, -2, -3), T(B::INT, three))))
		return false;

	if (!sim.addUpdate(
					"intVector",
					E(C<int>(3, 3, 3), T(B::INT, three)) + E("intVector", B::INT, three)))
		return false;

	return true;
}
