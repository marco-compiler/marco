#include <modelica/frontend/ast/ASTNode.h>

using namespace modelica;
using namespace modelica::frontend;

ASTNode::ASTNode(ASTNodeKind kind, SourcePosition location)
		: kind(std::move(kind)),
			location(std::move(location))
{
}

ASTNode::ASTNode(const ASTNode& other)
		: ASTNode(other.getKind(), getLocation())
{
}

ASTNode::ASTNode(ASTNode&& other) = default;

ASTNode::~ASTNode() = default;

ASTNode& ASTNode::operator=(const ASTNode& other)
{
	if (this != &other)
	{
		this->kind = other.kind;
		this->location = other.location;
	}

	return *this;
}

ASTNode& ASTNode::operator=(ASTNode&& other) = default;

namespace modelica::frontend
{
	void swap(ASTNode& first, ASTNode& second)
	{
		using std::swap;
		swap(first.kind, second.kind);
		swap(first.location, second.location);
	}
}

void ASTNode::dump() const
{
	dump(llvm::outs(), 0);
}

SourcePosition ASTNode::getLocation() const
{
	return location;
}

void ASTNode::setLocation(SourcePosition loc)
{
	this->location = loc;
}
