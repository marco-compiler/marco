#include "modelica/lowerer/SimType.hpp"

using namespace std;
using namespace modelica;
using namespace llvm;

void SimType::dump(raw_ostream& OS) const
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

void SimType::dumpCSyntax(StringRef name, raw_ostream& OS) const
{
	switch (builtinSimType)
	{
		case BultinSimTypes::INT:
			OS << "int";
			break;
		case BultinSimTypes::BOOL:
			OS << "bool";
			break;
		case BultinSimTypes::FLOAT:
			OS << "float";
			break;
	}
	OS << " " << name;
	for (auto dim : dimensions)
		OS << '[' << dim << ']';
}
