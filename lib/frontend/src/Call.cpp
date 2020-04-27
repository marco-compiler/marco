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
		: function(make_unique<Expression>(*other.function))
{
	for (const auto& exp : other.args)
		args.emplace_back(make_unique<Expression>(*exp));
}

Call& Call::operator=(const Call& other)
{
	if (this == &other)
		return *this;

	function = make_unique<Expression>(*other.function);
	args.clear();
	for (const auto& exp : other.args)
		args.emplace_back(make_unique<Expression>(*exp));
	return *this;
}
