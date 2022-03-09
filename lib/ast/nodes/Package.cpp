#include "marco/ast/AST.h"

using namespace marco::ast;

Package::Package(SourceRange location,
								 llvm::StringRef name,
								 llvm::ArrayRef<std::unique_ptr<Class>> innerClasses)
		: ASTNode(std::move(location)),
			name(name.str())
{
	for (const auto& cls : innerClasses)
		this->innerClasses.push_back(cls->clone());
}

Package::Package(const Package& other)
		: ASTNode(other),
			name(other.name)
{
	for (const auto& cls : other.innerClasses)
		this->innerClasses.push_back(cls->clone());
}

Package::Package(Package&& other) = default;

Package::~Package() = default;

Package& Package::operator=(const Package& other)
{
	Package result(other);
	swap(*this, result);
	return *this;
}

Package& Package::operator=(Package&& other) = default;

namespace marco::ast
{
	void swap(Package& first, Package& second)
	{
		swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

		using std::swap;
		impl::swap(first.innerClasses, second.innerClasses);
	}
}

void Package::print(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "package " << getName() << "\n";

	for (const auto& cls : innerClasses)
		cls->print(os, indents + 1);
}

llvm::StringRef Package::getName() const
{
	return name;
}

llvm::MutableArrayRef<std::unique_ptr<Class>> Package::getInnerClasses()
{
	return innerClasses;
}

llvm::ArrayRef<std::unique_ptr<Class>> Package::getInnerClasses() const
{
	return innerClasses;
}

size_t Package::size() const
{
	return innerClasses.size();
}

Package::iterator Package::begin()
{
	return innerClasses.begin();
}

Package::const_iterator Package::begin() const
{
	return innerClasses.begin();
}

Package::iterator Package::end()
{
	return innerClasses.end();
}

Package::const_iterator Package::end() const
{
	return innerClasses.end();
}
