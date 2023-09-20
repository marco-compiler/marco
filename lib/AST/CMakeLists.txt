set(AST_NODES
	Node/Algorithm.cpp
	Node/Annotation.cpp
	Node/ArrayConstant.cpp
	Node/ArrayDimension.cpp
	Node/ArrayGenerator.cpp
	Node/ArrayForGenerator.cpp
	Node/AssignmentStatement.cpp
	Node/ASTNode.cpp
	Node/BreakStatement.cpp
	Node/Call.cpp
	Node/Class.cpp
	Node/ComponentReference.cpp
	Node/ComponentReferenceEntry.cpp
	Node/EquationsBlock.cpp
	Node/ExpressionFunctionArgument.cpp
	Node/Model.cpp
	Node/Constant.cpp
	Node/Equation.cpp
	Node/Expression.cpp
	Node/ForEquation.cpp
	Node/ForIndex.cpp
	Node/ForStatement.cpp
	Node/Function.cpp
	Node/FunctionArgument.cpp
	Node/IfStatement.cpp
	Node/Induction.cpp
	Node/Member.cpp
	Node/Modification.cpp
	Node/NamedFunctionArgument.cpp
	Node/Operation.cpp
	Node/Package.cpp
	Node/Record.cpp
	Node/Reduction.cpp
	Node/ReductionFunctionArgument.cpp
	Node/ReturnStatement.cpp
	Node/Root.cpp
	Node/Statement.cpp
	Node/StatementsBlock.cpp
	Node/Subscript.cpp
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
