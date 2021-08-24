#include <marco/frontend/AST.h>

using namespace marco;
using namespace marco::frontend;

Statement::Statement(AssignmentStatement statement)
		: content(std::move(statement))
{
}

Statement::Statement(BreakStatement statement)
		: content(std::move(statement))
{
}

Statement::Statement(ForStatement statement)
		: content(std::move(statement))
{
}

Statement::Statement(IfStatement statement)
		: content(std::move(statement))
{
}

Statement::Statement(ReturnStatement statement)
		: content(std::move(statement))
{
}

Statement::Statement(WhenStatement statement)
		: content(std::move(statement))
{
}

Statement::Statement(WhileStatement statement)
		: content(std::move(statement))
{
}

Statement::Statement(const Statement& other)
		: content(other.content)
{
}

Statement::Statement(Statement&& other) = default;

Statement::~Statement() = default;

Statement& Statement::operator=(const Statement& other)
{
	Statement result(other);
	swap(*this, result);
	return *this;
}

Statement& Statement::operator=(Statement&& other) = default;

namespace marco::frontend
{
	void swap(Statement& first, Statement& second)
	{
		using std::swap;
		first.content = second.content;
	}
}

void Statement::print(llvm::raw_ostream &os, size_t indents) const
{
	visit([&os, indents](const auto& obj) {
		obj.print(os, indents);
	});
}

SourceRange Statement::getLocation() const
{
	return visit([](const auto& obj) {
		return obj.getLocation();
	});
}

Statement::assignments_iterator Statement::begin()
{
	return assignments_iterator(this, this);
}

Statement::assignments_const_iterator Statement::begin() const
{
	return assignments_const_iterator(this, this);
}

Statement::assignments_iterator Statement::end()
{
	return assignments_iterator(this, nullptr);
}

Statement::assignments_const_iterator Statement::end() const
{
	return assignments_const_iterator(this, nullptr);
}

AssignmentStatement::AssignmentStatement(SourceRange location,
																				 std::unique_ptr<Expression> destinations,
																				 std::unique_ptr<Expression> expression)
		: ASTNode(std::move(location)),
			destinations(std::move(destinations)),
			expression(std::move(expression))
{
	if (!this->destinations->isa<Tuple>())
	{
		auto type = this->destinations->getType();
		this->destinations = Expression::tuple(getLocation(), std::move(type), std::move(this->destinations));
	}

	assert(this->destinations->isa<Tuple>());
}

AssignmentStatement::AssignmentStatement(const AssignmentStatement& other)
		: ASTNode(other),
			destinations(other.destinations->clone()),
			expression(other.expression->clone())
{
	assert(destinations->isa<Tuple>());
}

AssignmentStatement::AssignmentStatement(AssignmentStatement&& other) = default;

AssignmentStatement::~AssignmentStatement() = default;

AssignmentStatement& AssignmentStatement::operator=(const AssignmentStatement& other)
{
	AssignmentStatement result(other);
	swap(*this, result);
	return *this;
}

AssignmentStatement& AssignmentStatement::operator=(AssignmentStatement&& other) = default;

namespace marco::frontend
{
	void swap(AssignmentStatement& first, AssignmentStatement& second)
	{
		swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

		using std::swap;
		swap(first.destinations, second.destinations);
		swap(first.expression, second.expression);
	}
}

void AssignmentStatement::print(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "destinations:\n";
	destinations->print(os, indents + 1);

	os.indent(indents);
	os << "assigned expression:\n";
	expression->print(os, indents + 1);
}

Expression* AssignmentStatement::getDestinations()
{
	return destinations.get();
}

const Expression* AssignmentStatement::getDestinations() const
{
	return destinations.get();
}

void AssignmentStatement::setDestinations(std::unique_ptr<Expression> dest)
{
	this->destinations = std::move(dest);

	if (!this->destinations->isa<Tuple>())
	{
		auto type = this->destinations->getType();
		this->destinations = Expression::tuple(getLocation(), std::move(type), std::move(this->destinations));
	}
}

Expression* AssignmentStatement::getExpression()
{
	return expression.get();
}

const Expression* AssignmentStatement::getExpression() const
{
	return expression.get();
}

IfStatement::IfStatement(SourceRange location, llvm::ArrayRef<Block> blocks)
		: ASTNode(std::move(location)),
			blocks(blocks.begin(), blocks.end())
{
	assert(!this->blocks.empty());
}

IfStatement::IfStatement(const IfStatement& other)
		: ASTNode(other),
			blocks(other.blocks.begin(), other.blocks.end())
{
}

IfStatement::IfStatement(IfStatement&& other) = default;

IfStatement::~IfStatement() = default;

IfStatement& IfStatement::operator=(const IfStatement& other)
{
	IfStatement result(other);
	swap(*this, result);
	return *this;
}

IfStatement& IfStatement::operator=(IfStatement&& other) = default;

namespace marco::frontend
{
	void swap(IfStatement& first, IfStatement& second)
	{
		swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

		using std::swap;
		swap(first.blocks, second.blocks);
	}
}

void IfStatement::print(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "if statement\n";

	for (const auto& block : blocks)
		block.print(os, indents + 1);
}

IfStatement::Block& IfStatement::operator[](size_t index)
{
	return getBlock(index);
}

const IfStatement::Block& IfStatement::operator[](size_t index) const
{
	return getBlock(index);
}

IfStatement::Block& IfStatement::getBlock(size_t index)
{
	assert(index < blocks.size());
	return blocks[index];
}

const IfStatement::Block& IfStatement::getBlock(size_t index) const
{
	assert(index < blocks.size());
	return blocks[index];
}

size_t IfStatement::size() const
{
	return blocks.size();
}

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

ForStatement::ForStatement(SourceRange location,
													 std::unique_ptr<Induction> induction,
                           llvm::ArrayRef<std::unique_ptr<Statement>> statements)
		: ASTNode(std::move(location)),
			induction(std::move(induction))
{
	for (const auto& statement : statements)
		this->statements.push_back(statement->clone());
}

ForStatement::ForStatement(const ForStatement& other)
		: ASTNode(other),
			induction(other.induction->clone()),
			breakCheckName(other.breakCheckName),
			returnCheckName(other.returnCheckName)
{
	for (const auto& statement : other.statements)
		this->statements.push_back(statement->clone());
}

ForStatement::ForStatement(ForStatement&& other) = default;

ForStatement::~ForStatement() = default;

ForStatement& ForStatement::operator=(const ForStatement& other)
{
	ForStatement result(other);
	swap(*this, result);
	return *this;
}

ForStatement& ForStatement::operator=(ForStatement&& other) = default;

namespace marco::frontend
{
	void swap(ForStatement& first, ForStatement& second)
	{
		swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

		using std::swap;
		swap(first.induction, second.induction);
		impl::swap(first.statements, second.statements);
		swap(first.breakCheckName, second.breakCheckName);
		swap(first.returnCheckName, second.returnCheckName);
	}
}

void ForStatement::print(llvm::raw_ostream& os, size_t indents) const
{
    os.indent(indents);
    os << "induction:\n";
    induction->print(os, indents + 1);

    os.indent(indents);
    os << "body:\n";

    for (const auto& statement : statements)
        statement->print(os, indents + 1);
}

Statement* ForStatement::operator[](size_t index)
{
	assert(index < statements.size());
	return statements[index].get();
}

const Statement* ForStatement::operator[](size_t index) const
{
	assert(index < statements.size());
	return statements[index].get();
}

llvm::StringRef ForStatement::getBreakCheckName() const
{
	return breakCheckName;
}

void ForStatement::setBreakCheckName(llvm::StringRef name)
{
	this->breakCheckName = name.str();
}

llvm::StringRef ForStatement::getReturnCheckName() const
{
	return returnCheckName;
}

void ForStatement::setReturnCheckName(llvm::StringRef name)
{
	this->returnCheckName = name.str();
}

Induction* ForStatement::getInduction()
{
	return induction.get();
}

const Induction* ForStatement::getInduction() const
{
	return induction.get();
}

llvm::MutableArrayRef<std::unique_ptr<Statement>> ForStatement::getBody()
{
	return statements;
}

llvm::ArrayRef<std::unique_ptr<Statement>> ForStatement::getBody() const
{
	return statements;
}

void ForStatement::setBody(llvm::ArrayRef<std::unique_ptr<Statement>> body)
{
	statements.clear();

	for (const auto& statement : body)
		statements.push_back(statement->clone());
}

size_t ForStatement::size() const
{
	return statements.size();
}

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

WhileStatement::WhileStatement(SourceRange location,
															 std::unique_ptr<Expression> condition,
                               llvm::ArrayRef<std::unique_ptr<Statement>> body)
		: ASTNode(std::move(location)),
			ConditionalBlock<Statement>(std::move(condition), body)
{
}

WhileStatement::WhileStatement(const WhileStatement& other)
		: ASTNode(other),
			ConditionalBlock<Statement>(other),
			breakCheckName(other.breakCheckName),
			returnCheckName(other.returnCheckName)
{
}

WhileStatement::WhileStatement(WhileStatement&& other) = default;

WhileStatement::~WhileStatement() = default;

WhileStatement& WhileStatement::operator=(const WhileStatement& other)
{
	WhileStatement result(other);
	swap(*this, result);
	return *this;
}

WhileStatement& WhileStatement::operator=(WhileStatement&& other) = default;

namespace marco::frontend
{
	void swap(WhileStatement& first, WhileStatement& second)
	{
		swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

		swap(static_cast<ConditionalBlock<Statement>&>(first),
				 static_cast<ConditionalBlock<Statement>&>(second));

		using std::swap;
		swap(first.breakCheckName, second.breakCheckName);
		swap(first.returnCheckName, second.returnCheckName);
	}
}

void WhileStatement::print(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "while:\n";

	os.indent(indents + 1);
	os << "condition:\n";
	getCondition()->print(os, indents + 2);

	os.indent(indents + 1);
	os << "body:\n";

	for (const auto& statement : getBody())
		statement->print(os, indents + 2);
}

Statement* WhileStatement::operator[](size_t index)
{
	assert(index < getBody().size());
	return getBody()[index].get();
}

const Statement* WhileStatement::operator[](size_t index) const
{
	assert(index < getBody().size());
	return getBody()[index].get();
}

llvm::StringRef WhileStatement::getBreakCheckName() const
{
	return breakCheckName;
}

void WhileStatement::setBreakCheckName(llvm::StringRef name)
{
	this->breakCheckName = name.str();
}

llvm::StringRef WhileStatement::getReturnCheckName() const
{
	return returnCheckName;
}

void WhileStatement::setReturnCheckName(llvm::StringRef name)
{
	this->returnCheckName = name.str();
}

WhenStatement::WhenStatement(SourceRange location,
														 std::unique_ptr<Expression> condition,
														 llvm::ArrayRef<std::unique_ptr<Statement>> body)
		: ASTNode(std::move(location)),
			ConditionalBlock<Statement>(std::move(condition), body)
{
}

WhenStatement::WhenStatement(const WhenStatement& other)
		: ASTNode(other),
			ConditionalBlock<Statement>(other)
{
}

WhenStatement::WhenStatement(WhenStatement&& other) = default;

WhenStatement::~WhenStatement() = default;

WhenStatement& WhenStatement::operator=(const WhenStatement& other)
{
	WhenStatement result(other);
	swap(*this, result);
	return *this;
}

WhenStatement& WhenStatement::operator=(WhenStatement&& other) = default;

namespace marco::frontend
{
	void swap(WhenStatement& first, WhenStatement& second)
	{
		swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

		using std::swap;
	}
}

void WhenStatement::print(llvm::raw_ostream& os, size_t indents) const
{
}

BreakStatement::BreakStatement(SourceRange location)
		: ASTNode(std::move(location))
{
}

BreakStatement::BreakStatement(const BreakStatement& other)
		: ASTNode(other)
{
}

BreakStatement::BreakStatement(BreakStatement&& other) = default;

BreakStatement::~BreakStatement() = default;

BreakStatement& BreakStatement::operator=(const BreakStatement& other)
{
	BreakStatement result(other);
	swap(*this, result);
	return *this;
}

BreakStatement& BreakStatement::operator=(BreakStatement&& other) = default;

namespace marco::frontend
{
	void swap(BreakStatement& first, BreakStatement& second)
	{
		swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

		using std::swap;
	}
}

void BreakStatement::print(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "break\n";
}

ReturnStatement::ReturnStatement(SourceRange location)
		: ASTNode(std::move(location))
{
}

ReturnStatement::ReturnStatement(const ReturnStatement& other)
		: ASTNode(other),
			returnCheckName(other.returnCheckName)
{
}

ReturnStatement::ReturnStatement(ReturnStatement&& other) = default;

ReturnStatement::~ReturnStatement() = default;

ReturnStatement& ReturnStatement::operator=(const ReturnStatement& other)
{
	ReturnStatement result(other);
	swap(*this, result);
	return *this;
}

ReturnStatement& ReturnStatement::operator=(ReturnStatement&& other) = default;

namespace marco::frontend
{
	void swap(ReturnStatement& first, ReturnStatement& second)
	{
		swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

		using std::swap;
		swap(first.returnCheckName, second.returnCheckName);
	}
}

void ReturnStatement::print(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "return\n";
}

llvm::StringRef ReturnStatement::getReturnCheckName() const
{
	return returnCheckName;
}

void ReturnStatement::setReturnCheckName(llvm::StringRef name)
{
	this->returnCheckName = name.str();
}
