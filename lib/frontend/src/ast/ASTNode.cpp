#include <modelica/frontend/ast/ASTNode.h>

using namespace modelica;
using namespace modelica::frontend;

ASTNode::ASTNode(SourcePosition location)
		: location(std::move(location))
{
}

ASTNode::ASTNode(const ASTNode& other) = default;

ASTNode::ASTNode(ASTNode&& other) = default;

ASTNode::~ASTNode() = default;

ASTNode& ASTNode::operator=(const ASTNode& other)
{
	if (this != &other)
	{
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
		swap(first.location, second.location);
	}
}

SourcePosition ASTNode::getLocation() const
{
	return location;
}
