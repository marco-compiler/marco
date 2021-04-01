#include <modelica/frontend/AST.h>

using namespace modelica;

Package::Package(
		SourcePosition location,
		std::string name,
		llvm::ArrayRef<ClassContainer> innerClasses)
		: location(std::move(location)),
			name(std::move(name))
{
	assert(!this->name.empty());

	for (const auto& cls : innerClasses)
		this->innerClasses.emplace_back(std::make_shared<ClassContainer>(cls));
}

void Package::dump() const { dump(llvm::outs(), 0); }

void Package::dump(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "package " << name << "\n";

	for (const auto& cls : innerClasses)
		cls->dump(os, indents + 1);
}

SourcePosition Package::getLocation() const
{
	return location;
}

const std::string& Package::getName() const
{
	return name;
}

Package::Container<ClassContainer>& Package::getInnerClasses()
{
	return innerClasses;
}

const Package::Container<ClassContainer>& Package::getInnerClasses() const
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
