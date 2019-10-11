#include "modelica/simulatorGenerator/Generator.hpp"

using namespace modelica;
using namespace llvm;

using E = SimExp;
using T = BultinSimTypes;

template<typename T>
using C = modelica::SimConst<T>;

bool makeSimulation(Simulation& sim)
{
	if (!sim.addVar("FloatConstant", E(C<float>(3.0F))))
		return false;
	if (!sim.addVar("FloatModifiable", E(C<float>(3.0F))))
		return false;

	if (!sim.addUpdate(
					"FloatModifiable",
					E("FloatModifiable", T::FLOAT) + E(C<float>(1.0F))))
		return false;
	return true;
}
