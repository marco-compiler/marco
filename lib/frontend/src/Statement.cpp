#include <modelica/frontend/Statement.hpp>

using namespace llvm;
using namespace modelica;
using namespace std;

AssignmentStatement::AssignmentStatement(
		SourcePosition location, Expression destination, Expression expression)
		: location(location),
			destinations(Tuple(location, move(destination))),
			expression(move(expression))
{
}

AssignmentStatement::AssignmentStatement(
		SourcePosition location, Tuple destinations, Expression expression)
		: location(move(location)),
			destinations(move(destinations)),
			expression(move(expression))
{
}

AssignmentStatement::AssignmentStatement(
		SourcePosition location, initializer_list<Expression> destinations, Expression expression)
		: location(location),
			destinations(Tuple(location, move(destinations))),
			expression(move(expression))
{
}

void AssignmentStatement::dump() const { dump(outs(), 0); }

void AssignmentStatement::dump(raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "destinations:\n";
	destinations.dump(os, indents + 1);

	os.indent(indents);
	os << "assigned expression:\n";
	expression.dump(os, indents + 1);
}

SourcePosition AssignmentStatement::getLocation() const
{
	return location;
}

Tuple& AssignmentStatement::getDestinations()
{
	return destinations;
}

const Tuple& AssignmentStatement::getDestinations() const
{
	return destinations;
}

void AssignmentStatement::setDestination(Expression dest)
{
	auto loc = dest.getLocation();
	destinations = Tuple(loc, move(dest));
}

void AssignmentStatement::setDestination(Tuple dest)
{
	destinations = move(dest);
}

Expression& AssignmentStatement::getExpression() { return expression; }

const Expression& AssignmentStatement::getExpression() const
{
	return expression;
}

IfStatement::IfStatement(SourcePosition location, llvm::ArrayRef<Block> blocks)
		: location(move(location)),
			blocks(blocks.begin(), blocks.end())
{
	assert(!this->blocks.empty());
}

IfStatement::Block& IfStatement::operator[](size_t index)
{
	assert(index < blocks.size());
	return blocks[index];
}

const IfStatement::Block& IfStatement::operator[](size_t index) const
{
	assert(index < blocks.size());
	return blocks[index];
}

void IfStatement::dump() const { dump(outs(), 0); }

void IfStatement::dump(raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "if statement\n";

	for (const auto& block : blocks)
		block.dump(os, indents + 1);
}

SourcePosition IfStatement::getLocation() const
{
	return location;
}

size_t IfStatement::size() const { return blocks.size(); }

IfStatement::blocks_iterator IfStatement::begin()
{
	return blocks.begin();
}

IfStatement::blocks_const_iterator IfStatement::begin() const
{
	return blocks.begin();
}

IfStatement::blocks_iterator IfStatement::end()
{
	return blocks.end();
}

IfStatement::blocks_const_iterator IfStatement::end() const
{
	return blocks.end();
}

ForStatement::ForStatement(
		SourcePosition location, Induction induction, llvm::ArrayRef<Statement> statements)
		: location(move(location)),
			induction(move(induction))
{
	for (const auto& statement : statements)
		this->statements.push_back(std::make_unique<Statement>(statement));
}

ForStatement::ForStatement(const ForStatement& other)
		: location(other.location),
			induction(other.induction)
{
	statements.clear();

	for (const auto& statement : other.statements)
		statements.push_back(std::make_unique<Statement>(*statement));
}

ForStatement& ForStatement::operator=(const ForStatement& other)
{
	if (this == &other)
		return *this;

	location = other.location;
	induction = other.induction;
	statements.clear();

	for (const auto& statement : other.statements)
		statements.push_back(std::make_unique<Statement>(*statement));

	return *this;
}

Statement& ForStatement::operator[](size_t index)
{
	assert(index < statements.size());
	return *statements[index];
}

const Statement& ForStatement::operator[](size_t index) const
{
	assert(index < statements.size());
	return *statements[index];
}

void ForStatement::dump() const { dump(outs(), 0); }

void ForStatement::dump(raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "induction:\n";
	induction.dump(os, indents + 1);

	os.indent(indents);
	os << "body:\n";

	for (const auto& statement : statements)
		statement->dump(os, indents + 1);
}

SourcePosition ForStatement::getLocation() const
{
	return location;
}

Induction& ForStatement::getInduction() { return induction; }

const Induction& ForStatement::getInduction() const { return induction; }

size_t ForStatement::size() const { return statements.size(); }

ForStatement::statements_iterator ForStatement::begin()
{
	return statements.begin();
}

ForStatement::statements_const_iterator ForStatement::begin() const
{
	return statements.begin();
}

ForStatement::statements_iterator ForStatement::end()
{
	return statements.end();
}

ForStatement::statements_const_iterator ForStatement::end() const
{
	return statements.end();
}

WhileStatement::WhileStatement(
		SourcePosition location, Expression condition, llvm::ArrayRef<Statement> body)
		: ConditionalBlock<Statement>(move(condition), move(body)),
			location(move(location))
{
}

void WhileStatement::dump() const { dump(outs(), 0); }

void WhileStatement::dump(raw_ostream& os, size_t indents) const
{
}

SourcePosition WhileStatement::getLocation() const
{
	return location;
}

WhenStatement::WhenStatement(
		SourcePosition location, Expression condition, llvm::ArrayRef<Statement> body)
		: ConditionalBlock<Statement>(move(condition), move(body)),
			location(move(location))
{
}

void WhenStatement::dump() const { dump(outs(), 0); }

void WhenStatement::dump(raw_ostream& os, size_t indents) const
{
}

SourcePosition WhenStatement::getLocation() const
{
	return location;
}

BreakStatement::BreakStatement(SourcePosition location)
		: location(move(location))
{
}

void BreakStatement::dump() const { dump(outs(), 0); }

void BreakStatement::dump(raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "break\n";
}

SourcePosition BreakStatement::getLocation() const
{
	return location;
}

ReturnStatement::ReturnStatement(SourcePosition location)
		: location(move(location))
{
}

void ReturnStatement::dump() const { dump(outs(), 0); }

void ReturnStatement::dump(raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "return\n";
}

SourcePosition ReturnStatement::getLocation() const
{
	return location;
}

Statement::Statement(AssignmentStatement statement)
		: content(move(statement))
{
}

Statement::Statement(IfStatement statement)
		: content(move(statement))
{
}

Statement::Statement(ForStatement statement)
		: content(move(statement))
{
}

Statement::Statement(WhileStatement statement)
		: content(move(statement))
{
}

Statement::Statement(WhenStatement statement)
		: content(move(statement))
{
}

Statement::Statement(BreakStatement statement)
		: content(move(statement))
{
}

Statement::Statement(ReturnStatement statement)
		: content(move(statement))
{
}

void Statement::dump() const { dump(outs(), 0); }

void Statement::dump(raw_ostream& os, size_t indents) const
{
	visit([&](const auto& statement) { statement.dump(os, indents); });
}

Statement::assignments_iterator Statement::begin() { return assignments_iterator(this, this); }

Statement::assignments_const_iterator Statement::begin() const
{
	return assignments_const_iterator(this, this);
}

Statement::assignments_iterator Statement::end() { return assignments_iterator(this, nullptr); }

Statement::assignments_const_iterator Statement::end() const
{
	return assignments_const_iterator(this, nullptr);
}
