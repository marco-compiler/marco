#include "Generator.hpp"

using namespace modelica;
using namespace llvm;

using E = SimExp;
using T = BultinSimTypes;

template<typename T>
using C = modelica::SimConst<T>;

bool makeSimulation(Simulation& sim)
{
	if (!sim.addVar("X", E(C<int>(3))))
		return false;
	if (!sim.addVar("Y", E(C<int>(6))))
		return false;

	if (!sim.addUpdate("X", E(C<int>(3)) + E("Y", T::INT)))
		return false;

	return true;
}
