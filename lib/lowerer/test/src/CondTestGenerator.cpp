#include "modelica/simulatorGenerator/Generator.hpp"

using namespace modelica;
using namespace llvm;

using E = SimExp;
using T = BultinSimTypes;

template<typename T>
using C = modelica::SimConst<T>;

bool makeSimulation(Lowerer& sim)
{
	if (!sim.addVar("leftHand", E(C<int>(3))))
		return false;
	if (!sim.addVar("rightHand", E(C<int>(4))))
		return false;

	if (!sim.addVar("res", E(C<int>(0))))
		return false;

	if (!sim.addUpdate(
					"res",
					E::cond(
							E("leftHand", T::INT) > E("rightHand", T::INT),
							E(C<int>(1)),
							E(C<int>(9)))))
		return false;
	return true;
}
