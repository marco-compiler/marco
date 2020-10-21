#include <iostream>
#include <modelica/frontend/Type.hpp>

using namespace std;
using namespace llvm;
using namespace modelica;

namespace modelica
{
	raw_ostream& operator<<(raw_ostream& stream, const BuiltinType& obj)
	{
		if (obj == BuiltinType::None)
			stream << "None";
		else if (obj == BuiltinType::Integer)
			stream << "Integer";
		else if (obj == BuiltinType::Float)
			stream << "Float";
		else if (obj == BuiltinType::String)
			stream << "String";
		else if (obj == BuiltinType::Boolean)
			stream << "Boolean";
		else if (obj == BuiltinType::Unknown)
			stream << "Unknown";

		return stream;
	}
}	 // namespace modelica

void Type::dump(raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << type;

	if (!isScalar())
		for (size_t dim : dimensions)
		{
			os << dim;
			os << " ";
		}
}
