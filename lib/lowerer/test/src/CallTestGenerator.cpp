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
	if (!sim.addVar("Int", E(C<int>(2))))
		return false;

	if (!sim.addVar("IntConst", E(C<int>(3))))
		return false;

	if (!sim.addUpdate(
					"Int",
					E(SimCall(
							"mult",
							{ E(C<int>(2)), E("IntConst", BultinSimTypes::INT) },
							T(BultinSimTypes::INT)))))
		return false;
	return true;
}
