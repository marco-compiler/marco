#include "modelica/lowerer/SimType.hpp"

using namespace std;
using namespace modelica;
using namespace llvm;

void SimType::dump(llvm::raw_ostream& OS) const
{
	switch (builtinSimType)
	{
		case BultinSimTypes::INT:
			OS << "INT";
			break;
		case BultinSimTypes::BOOL:
			OS << "BOOL";
			break;
		case BultinSimTypes::FLOAT:
			OS << "FLOAT";
			break;
	}
	OS << '[';

	for (auto iter = begin(dimensions); iter != end(dimensions); iter++)
	{
		OS << *iter;
		if (iter != end(dimensions))
			OS << " ,";
	}
	OS << ']';
}
