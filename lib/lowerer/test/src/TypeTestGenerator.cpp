#include "modelica/simulatorGenerator/Generator.hpp"

using namespace modelica;
using namespace llvm;

using E = SimExp;
using T = BultinSimTypes;

template<typename T>
using C = modelica::SimConst<T>;

bool makeSimulation(Lowerer& sim)
{
	if (!sim.addVar("IntModifiable", E(C<int>(3))))
		return false;
	if (!sim.addVar("IntConstant", E(C<int>(6))))
		return false;
	if (!sim.addVar("BoolConstant", E(C<bool>(0))))
		return false;

	if (!sim.addUpdate("IntModifiable", E(C<int>(3)) + E("IntConstant", T::INT)))
		return false;
	if (!sim.addUpdate("BoolConstant", !E("IntModifiable", T::BOOL)))
		return false;

	return true;
}
