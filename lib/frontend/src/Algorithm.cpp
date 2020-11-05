#include <modelica/frontend/Algorithm.hpp>

using namespace std;
using namespace llvm;
using namespace modelica;

Algorithm::Algorithm(ArrayRef<Statement> statements)
		: statements(
					iterator_range<ArrayRef<Statement>::iterator>(move(statements)))
{
}

void Algorithm::dump() const { dump(outs(), 0); }

void Algorithm::dump(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "algorithm:\n";

	for (const auto& statement : statements)
		statement.dump(os, indents + 1);
}

SmallVectorImpl<Statement>& Algorithm::getStatements() { return statements; }

const SmallVectorImpl<Statement>& Algorithm::getStatements() const
{
	return statements;
}
