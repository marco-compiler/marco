#include <modelica/frontend/AST.h>

using namespace modelica::frontend;

ClassContainer::ClassContainer(Class cls) : content(cls) {}

ClassContainer::ClassContainer(Function function) : content(function) {}

ClassContainer::ClassContainer(Package package) : content(package) {}

ClassContainer::ClassContainer(Record record) : content(record) {}

void ClassContainer::dump() const { dump(llvm::outs(), 0); }

void ClassContainer::dump(llvm::raw_ostream& os, size_t indents) const
{
	visit([&](const auto& obj) { obj.dump(os, indents); });
}
