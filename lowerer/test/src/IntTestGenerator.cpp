#include "Generator.hpp"

using namespace modelica;
using namespace llvm;

bool makeSimulation(Simulation& sim)
{
	if (!sim.addVar("X", SimExp(SimConst(3))))
		return false;
	if (!sim.addVar("Y", SimExp(SimConst(6))))
		return false;

	if (!sim.addUpdate(
					"X", SimExp(SimConst(3)) + SimExp("Y", BultinSimTypes::INT)))
		return false;

	return true;
}
