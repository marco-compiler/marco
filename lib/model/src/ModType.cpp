#include "modelica/model/ModType.hpp"

using namespace std;
using namespace modelica;
using namespace llvm;

void ModType::dump(raw_ostream& OS) const
{
	switch (builtinModType)
	{
		case BultinModTypes::INT:
			OS << "INT";
			break;
		case BultinModTypes::BOOL:
			OS << "BOOL";
			break;
		case BultinModTypes::FLOAT:
			OS << "FLOAT";
			break;
	}
	OS << '[';

	for (auto iter = begin(dimensions); iter != end(dimensions); iter++)
	{
		OS << *iter;
		if (next(iter) != end(dimensions))
			OS << " ,";
	}
	OS << ']';
}

void ModType::dumpCSyntax(
		StringRef name, bool useDoubles, raw_ostream& OS) const
{
	switch (builtinModType)
	{
		case BultinModTypes::INT:
			OS << "int";
			break;
		case BultinModTypes::BOOL:
			OS << "bool";
			break;
		case BultinModTypes::FLOAT:
			OS << (useDoubles ? "double" : "float");
			break;
	}
	OS << " " << name;
	for (auto dim : dimensions)
		OS << '[' << dim << ']';
}
