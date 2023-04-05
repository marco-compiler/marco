set(AST_NODES
	Node/Algorithm.cpp
	Node/Annotation.cpp
	Node/Array.cpp
	Node/ArrayDimension.cpp
	Node/AssignmentStatement.cpp
	Node/ASTNode.cpp
	Node/BreakStatement.cpp
	Node/Call.cpp
	Node/Class.cpp
	Node/EquationsBlock.cpp
	Node/Model.cpp
	Node/Constant.cpp
	Node/Equation.cpp
	Node/Expression.cpp
	Node/ForEquation.cpp
	Node/ForStatement.cpp
	Node/Function.cpp
	Node/IfStatement.cpp
	Node/Induction.cpp
	Node/Member.cpp
	Node/Modification.cpp
	Node/Operation.cpp
	Node/Package.cpp
	Node/Record.cpp
	Node/ReferenceAccess.cpp
	Node/ReturnStatement.cpp
	Node/Root.cpp
	Node/Statement.cpp
	Node/StatementsBlock.cpp
	Node/Tuple.cpp
	Node/Type.cpp
	Node/TypePrefix.cpp
	Node/WhenStatement.cpp
	Node/WhileStatement.cpp)

set(SOURCES
	${AST_NODES})

marco_add_library(ast
	${SOURCES}

	LINK_LIBS
	PUBLIC marco::diagnostic

	LINK_COMPONENTS
	Core
	Support)
