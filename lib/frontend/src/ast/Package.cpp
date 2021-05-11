#include <modelica/frontend/AST.h>

using namespace modelica::frontend;

Package::Package(SourcePosition location,
								 llvm::StringRef name,
								 llvm::ArrayRef<std::unique_ptr<Class>> innerClasses)
		: ClassCRTP<Package>(ASTNodeKind::CLASS_PACKAGE, std::move(location), std::move(name))
{
	for (const auto& cls : innerClasses)
		this->innerClasses.push_back(cls->cloneClass());
}

Package::Package(const Package& other)
		: ClassCRTP<Package>(static_cast<ClassCRTP<Package>&>(*this))
{
	for (const auto& cls : other.innerClasses)
		this->innerClasses.push_back(cls->cloneClass());
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

namespace modelica::frontend
{
	void swap(Package& first, Package& second)
	{
		swap(static_cast<impl::ClassCRTP<Package>&>(first),
				 static_cast<impl::ClassCRTP<Package>&>(second));

		using std::swap;
		impl::swap(first.innerClasses, second.innerClasses);
	}
}

void Package::dump(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "package " << getName() << "\n";

	for (const auto& cls : innerClasses)
		cls->dump(os, indents + 1);
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
