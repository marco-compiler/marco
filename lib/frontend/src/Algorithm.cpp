#include <modelica/frontend/Algorithm.hpp>

using namespace llvm;
using namespace modelica;
using namespace std;

Algorithm::Algorithm(ArrayRef<Statement> statements)
		: statements(statements.begin(), statements.end())
{
}

void Algorithm::dump() const { dump(outs(), 0); }

void Algorithm::dump(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "algorithm\n";

	for (const auto& statement : statements)
		statement.visit([&](const auto& obj) { obj.dump(os, indents + 1); });
}

SmallVectorImpl<Statement>& Algorithm::getStatements() { return statements; }

const SmallVectorImpl<Statement>& Algorithm::getStatements() const
{
	return statements;
}
