#include <marco/frontend/ast/ASTNode.h>

using namespace marco;
using namespace marco::frontend;

ASTNode::ASTNode(SourceRange location)
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

namespace marco::frontend
{
	void swap(ASTNode& first, ASTNode& second)
	{
		using std::swap;
		swap(first.location, second.location);
	}
}

SourceRange ASTNode::getLocation() const
{
	return location;
}
