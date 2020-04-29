#include "modelica/frontend/Call.hpp"

#include <memory>

#include "modelica/frontend/Expression.hpp"
#include "modelica/utils/IRange.hpp"

using namespace modelica;
using namespace llvm;
using namespace std;

[[nodiscard]] bool Call::operator==(const Call& other) const
{
	if (argumentsCount() != other.argumentsCount())
		return false;
	if (*function != *other.function)
		return false;

	for (auto i : irange(argumentsCount()))
		if (*args[i] != other[i])
			return false;

	return true;
}

Call::Call(const Call& other)
		: function(std::make_unique<Expression>(*other.function))
{
	for (const auto& exp : other.args)
		args.emplace_back(std::make_unique<Expression>(*exp));
}

Call& Call::operator=(const Call& other)
{
	if (this == &other)
		return *this;

	function = std::make_unique<Expression>(*other.function);
	args.clear();
	for (const auto& exp : other.args)
		args.emplace_back(std::make_unique<Expression>(*exp));
	return *this;
}

void Call::dump(llvm::raw_ostream& OS, size_t indentLevel) const
{
	OS.indent(indentLevel);
	OS << "call:\n";
	function->dump(OS, indentLevel + 1);
	OS << '\n';
	for (const auto& exp : args)
	{
		exp->dump(OS, indentLevel + 1);
		OS << '\n';
	}
}
