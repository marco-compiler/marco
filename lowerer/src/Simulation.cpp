#include "modelica/lowerer/Simulation.hpp"

using namespace std;
using namespace modelica;

void Simulation::lower() {}

void Simulation::dump(llvm::raw_ostream& OS) const
{
	auto const dumpEquation = [&OS](const auto& couple) {
		OS << couple.first().data();
		OS << " = ";
		couple.second.dump(OS);
		OS << "\n";
	};

	OS << "Init:\n";
	std::for_each(variables.begin(), variables.end(), dumpEquation);

	OS << "Update:\n";
	std::for_each(updates.begin(), updates.end(), dumpEquation);
}
