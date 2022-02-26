#include "marco/AST/AST.h"
#include <numeric>

using namespace marco::ast;

Call::Call(SourceRange location,
					 Type type,
					 std::unique_ptr<Expression> function,
					 llvm::ArrayRef<std::unique_ptr<Expression>> args)
		: ASTNode(std::move(location)),
			type(std::move(type)),
			function(std::move(function))
{
	for (const auto& arg : args)
		this->args.push_back(arg->clone());
}

Call::Call(const Call& other)
		: ASTNode(other),
			type(other.type),
			function(other.function->clone())
{
	for (const auto& arg : other.args)
		this->args.push_back(arg->clone());
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

namespace marco::ast
{
	void swap(Call& first, Call& second)
	{
		swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

		using std::swap;
		swap(first.type, second.type);
		swap(first.function, second.function);
		impl::swap(first.args, second.args);
	}
}

void Call::print(llvm::raw_ostream& os, size_t indents) const
{
	function->print(os, indents + 1);

	for (const auto& arg : *this)
		arg->print(os, indents + 1);
}

bool Call::isLValue() const
{
	return false;
}

bool Call::operator==(const Call& other) const
{
	if (type != other.type)
		return false;

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
	return getArg(index);
}

const Expression* Call::operator[](size_t index) const
{
	return getArg(index);
}

Type& Call::getType()
{
	return type;
}

const Type& Call::getType() const
{
	return type;
}

void Call::setType(Type tp)
{
	type = std::move(tp);
}

Expression* Call::getFunction()
{
	return function.get();
}

const Expression* Call::getFunction() const
{
	return function.get();
}

Expression* Call::getArg(size_t index)
{
	assert(index < argumentsCount());
	return args[index].get();
}

const Expression* Call::getArg(size_t index) const
{
	assert(index < argumentsCount());
	return args[index].get();
}

llvm::MutableArrayRef<std::unique_ptr<Expression>> Call::getArgs()
{
	return args;
}

llvm::ArrayRef<std::unique_ptr<Expression>> Call::getArgs() const
{
	return args;
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

namespace marco::ast
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
