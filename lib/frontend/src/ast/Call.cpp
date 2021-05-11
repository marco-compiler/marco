#include <modelica/frontend/AST.h>
#include <numeric>

using namespace modelica::frontend;

Call::Call(SourcePosition location,
					 std::unique_ptr<ReferenceAccess> function,
					 llvm::ArrayRef<std::unique_ptr<Expression>> args,
					 Type type)
		: ExpressionCRTP<Call>(
					ASTNodeKind::EXPRESSION_CALL, std::move(location), std::move(type)),
			function(std::move(function))
{
	for (const auto& arg : args)
		this->args.push_back(arg->cloneExpression());
}

Call::Call(const Call& other)
		: ExpressionCRTP<Call>(static_cast<ExpressionCRTP<Call>&>(*this)),
			function(other.function->clone())
{
	for (const auto& arg : other.args)
		this->args.push_back(arg->cloneExpression());
}

Call::Call(Call&& other) = default;

Call::~Call() = default;

Call& Call::operator=(const Call& other)
{
	Call result(other);
	swap(*this, result);
	return *this;
}

Call& Call::operator=(Call&& other) = default;

namespace modelica::frontend
{
	void swap(Call& first, Call& second)
	{
		swap(static_cast<impl::ExpressionCRTP<Call>&>(first),
				 static_cast<impl::ExpressionCRTP<Call>&>(second));

		std::swap(first.function, second.function);
		impl::swap(first.args, second.args);
	}
}

void Call::dump(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "call\n";

	os.indent(indents);
	os << "type: ";
	getType().dump(os);
	os << "\n";

	function->dump(os, indents + 1);

	for (const auto& arg : *this)
		arg->dump(os, indents + 1);
}

bool Call::isLValue() const
{
	return false;
}

bool Call::operator==(const Call& other) const
{
	if (argumentsCount() != other.argumentsCount())
		return false;

	if (*function != *other.function)
		return false;

	if (args.size() != other.args.size())
		return false;

	auto pairs = llvm::zip(args, other.args);
	return std::all_of(pairs.begin(), pairs.end(),
										 [](const auto& pair)
										 {
											 const auto& [x, y] = pair;
											 return *x == *y;
										 });
}

bool Call::operator!=(const Call& other) const
{
	return !(*this == other);
}

Expression* Call::operator[](size_t index)
{
	assert(index < argumentsCount());
	return args[index].get();
}

const Expression* Call::operator[](size_t index) const
{
	assert(index < argumentsCount());
	return args[index].get();
}

ReferenceAccess* Call::getFunction()
{
	return function.get();
}

const ReferenceAccess* Call::getFunction() const
{
	return function.get();
}

size_t Call::argumentsCount() const
{
	return args.size();
}

Call::args_iterator Call::begin()
{
	return args.begin();
}

Call::args_const_iterator Call::begin() const
{
	return args.begin();
}

Call::args_iterator Call::end()
{
	return args.end();
}

Call::args_const_iterator Call::end() const
{
	return args.end();
}

namespace modelica::frontend
{
	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Call& obj)
	{
		return stream << toString(obj);
	}

	std::string toString(const Call& obj)
	{
		return toString(*obj.getFunction()) + "(" +
					 accumulate(
							 obj.begin(), obj.end(), std::string(),
							 [](const std::string& result, const std::unique_ptr<Expression>& argument) {
								 std::string str = toString(*argument);
								 return result.empty() ? str : result + "," + str;
							 }) +
					 ")";
	}
}
