#include <modelica/frontend/AST.h>

using namespace modelica::frontend;

Statement::Statement(ASTNodeKind kind, SourcePosition location)
		: ASTNodeCRTP<Statement>(kind, std::move(location))
{
}

Statement::Statement(const Statement& other)
		: ASTNodeCRTP<Statement>(static_cast<ASTNodeCRTP<Statement>&>(*this))
{
}

Statement::Statement(Statement&& other) = default;

Statement::~Statement() = default;

Statement& Statement::operator=(const Statement& other)
{
	if (this != &other)
	{
		static_cast<ASTNodeCRTP<Statement>&>(*this) = static_cast<const ASTNodeCRTP<Statement>&>(other);
	}

	return *this;
}

Statement& Statement::operator=(Statement&& other) = default;

namespace modelica::frontend
{
	void swap(Statement& first, Statement& second)
	{
		swap(static_cast<impl::ASTNodeCRTP<Statement>&>(first),
				 static_cast<impl::ASTNodeCRTP<Statement>&>(second));

		using std::swap;
	}
}

Statement::assignments_iterator Statement::assignmentsBegin()
{
	return assignments_iterator(this, this);
}

Statement::assignments_const_iterator Statement::assignmentsBegin() const
{
	return assignments_const_iterator(this, this);
}

Statement::assignments_iterator Statement::assignmentsEnd()
{
	return assignments_iterator(this, nullptr);
}

Statement::assignments_const_iterator Statement::assignmentsEnd() const
{
	return assignments_const_iterator(this, nullptr);
}


AssignmentStatement::AssignmentStatement(SourcePosition location,
																				 std::unique_ptr<Expression> destination,
																				 std::unique_ptr<Expression> expression)
		: StatementCRTP<AssignmentStatement>(ASTNodeKind::STATEMENT_ASSIGNMENT, std::move(location)),
			destinations(std::make_unique<Tuple>(location, std::move(destination))),
			expression(std::move(expression))
{
}

AssignmentStatement::AssignmentStatement(SourcePosition location,
																				 std::unique_ptr<Tuple> destinations,
																				 std::unique_ptr<Expression> expression)
		: StatementCRTP<AssignmentStatement>(ASTNodeKind::STATEMENT_ASSIGNMENT, std::move(location)),
			destinations(std::move(destinations)),
			expression(std::move(expression))
{
}

AssignmentStatement::AssignmentStatement(const AssignmentStatement& other)
		: StatementCRTP<AssignmentStatement>(static_cast<StatementCRTP<AssignmentStatement>&>(*this)),
			destinations(other.destinations->clone()),
			expression(other.expression->cloneExpression())
{
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

namespace modelica::frontend
{
	void swap(AssignmentStatement& first, AssignmentStatement& second)
	{
		swap(static_cast<impl::StatementCRTP<AssignmentStatement>&>(first),
				 static_cast<impl::StatementCRTP<AssignmentStatement>&>(second));

		using std::swap;
		swap(first.destinations, second.destinations);
		swap(first.expression, second.expression);
	}
}

void AssignmentStatement::dump(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "destinations:\n";
	destinations->dump(os, indents + 1);

	os.indent(indents);
	os << "assigned expression:\n";
	expression->dump(os, indents + 1);
}

Tuple* AssignmentStatement::getDestinations()
{
	return destinations.get();
}

const Tuple* AssignmentStatement::getDestinations() const
{
	return destinations.get();
}

void AssignmentStatement::setDestinations(Expression* dest)
{
	this->destinations = std::make_unique<Tuple>(dest->getLocation(), dest->cloneExpression());
}

void AssignmentStatement::setDestinations(Tuple* dest)
{
	this->destinations = dest->clone();
}

Expression* AssignmentStatement::getExpression()
{
	return expression.get();
}

const Expression* AssignmentStatement::getExpression() const
{
	return expression.get();
}

IfStatement::IfStatement(SourcePosition location, llvm::ArrayRef<Block> blocks)
		: StatementCRTP<IfStatement>(ASTNodeKind::STATEMENT_IF, std::move(location)),
			blocks(blocks.begin(), blocks.end())
{
	assert(!this->blocks.empty());
}

IfStatement::IfStatement(const IfStatement& other)
		: StatementCRTP<IfStatement>(static_cast<StatementCRTP<IfStatement>&>(*this)),
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

namespace modelica::frontend
{
	void swap(IfStatement& first, IfStatement& second)
	{
		swap(static_cast<impl::StatementCRTP<IfStatement>&>(first),
				 static_cast<impl::StatementCRTP<IfStatement>&>(second));

		using std::swap;
		swap(first.blocks, second.blocks);
	}
}

void IfStatement::dump(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "if statement\n";

	for (const auto& block : blocks)
		block.dump(os, indents + 1);
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

ForStatement::ForStatement(SourcePosition location,
                           std::unique_ptr<Induction>& induction,
                           llvm::ArrayRef<std::unique_ptr<Statement>> statements)
		: StatementCRTP<ForStatement>(ASTNodeKind::STATEMENT_FOR, std::move(location)),
			induction(induction->clone())
{
	for (const auto& statement : statements)
		this->statements.push_back(statement->cloneStatement());
}

ForStatement::ForStatement(const ForStatement& other)
		: StatementCRTP<ForStatement>(static_cast<StatementCRTP<ForStatement>&>(*this)),
			induction(other.induction->clone()),
			breakCheckName(other.breakCheckName),
			returnCheckName(other.returnCheckName)
{
	for (const auto& statement : other.statements)
		this->statements.push_back(statement->cloneStatement());
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

namespace modelica::frontend
{
    void swap(ForStatement& first, ForStatement& second)
    {
        swap(static_cast<impl::StatementCRTP<ForStatement>&>(first),
             static_cast<impl::StatementCRTP<ForStatement>&>(second));

        using std::swap;
        swap(first.induction, second.induction);
        swap(first.statements, second.statements);
        swap(first.breakCheckName, second.breakCheckName);
        swap(first.returnCheckName, second.returnCheckName);
    }
}

void ForStatement::dump(llvm::raw_ostream& os, size_t indents) const
{
    os.indent(indents);
    os << "induction:\n";
    induction->dump(os, indents + 1);

    os.indent(indents);
    os << "body:\n";

    for (const auto& statement : statements)
        statement->dump(os, indents + 1);
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

WhileStatement::WhileStatement(SourcePosition location,
                               std::unique_ptr<Expression>& condition,
                               llvm::ArrayRef<std::unique_ptr<Statement>> body)
		: StatementCRTP<WhileStatement>(ASTNodeKind::STATEMENT_WHILE, std::move(location)),
			ConditionalBlock<Statement>(condition, body)
{
}

WhileStatement::WhileStatement(const WhileStatement& other)
		: StatementCRTP<WhileStatement>(static_cast<StatementCRTP<WhileStatement>&>(*this)),
			ConditionalBlock<Statement>(static_cast<ConditionalBlock<Statement>&>(*this))
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

namespace modelica::frontend
{
	void swap(WhileStatement& first, WhileStatement& second)
	{
		swap(static_cast<impl::StatementCRTP<WhileStatement>&>(first),
				 static_cast<impl::StatementCRTP<WhileStatement>&>(second));

		swap(static_cast<ConditionalBlock<Statement>&>(first),
				 static_cast<ConditionalBlock<Statement>&>(second));

		using std::swap;
		swap(first.breakCheckName, second.breakCheckName);
		swap(first.returnCheckName, second.returnCheckName);
	}
}

void WhileStatement::dump(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "while:\n";

	os.indent(indents + 1);
	os << "condition:\n";
	getCondition()->dump(os, indents + 2);

	os.indent(indents + 1);
	os << "body:\n";

	for (const auto& statement : getBody())
		statement->dump(os, indents + 2);
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

WhenStatement::WhenStatement(SourcePosition location,
                               std::unique_ptr<Expression>& condition,
                               llvm::ArrayRef<std::unique_ptr<Statement>> body)
        : StatementCRTP<WhenStatement>(ASTNodeKind::STATEMENT_WHEN, std::move(location)),
          ConditionalBlock<Statement>(condition, body)
{
}

WhenStatement::WhenStatement(const WhenStatement& other)
        : StatementCRTP<WhenStatement>(static_cast<StatementCRTP<WhenStatement>&>(*this)),
          ConditionalBlock<Statement>(static_cast<ConditionalBlock<Statement>&>(*this))
{
}

void WhenStatement::dump(llvm::raw_ostream& os, size_t indents) const
{
}

BreakStatement::BreakStatement(SourcePosition location)
        : StatementCRTP<BreakStatement>(ASTNodeKind::STATEMENT_BREAK, std::move(location))
{
}

BreakStatement::BreakStatement(const BreakStatement& other)
		: StatementCRTP<BreakStatement>(static_cast<StatementCRTP<BreakStatement>&>(*this))
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

namespace modelica::frontend
{
	void swap(BreakStatement& first, BreakStatement& second)
	{
		swap(static_cast<impl::StatementCRTP<BreakStatement>&>(first),
				 static_cast<impl::StatementCRTP<BreakStatement>&>(second));

		using std::swap;
	}
}

void BreakStatement::dump(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "break\n";
}

ReturnStatement::ReturnStatement(SourcePosition location)
		: StatementCRTP<ReturnStatement>(ASTNodeKind::STATEMENT_RETURN, std::move(location))
{
}

ReturnStatement::ReturnStatement(const ReturnStatement& other)
		: StatementCRTP<ReturnStatement>(static_cast<StatementCRTP<ReturnStatement>&>(*this)),
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

namespace modelica::frontend
{
	void swap(ReturnStatement& first, ReturnStatement& second)
	{
		swap(static_cast<impl::StatementCRTP<ReturnStatement>&>(first),
				 static_cast<impl::StatementCRTP<ReturnStatement>&>(second));

		using std::swap;
		swap(first.returnCheckName, second.returnCheckName);
	}
}

void ReturnStatement::dump(llvm::raw_ostream& os, size_t indents) const
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
