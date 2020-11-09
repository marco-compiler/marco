modelicaAddLibrary(
		frontend SHARED
		src/Algorithm.cpp
		src/Call.cpp
		src/Class.cpp
        src/Constant.cpp
        src/ConstantFolder.cpp
        src/Equation.cpp
        src/Expression.cpp
        src/ForEquation.cpp
        src/LexerStateMachine.cpp
        src/Member.cpp
        src/Parser.cpp
        src/ParserErrors.cpp
        src/ReferenceAccess.cpp
        src/Statement.cpp
        src/SymbolTable.cpp
        src/Tuple.cpp
        src/Type.cpp
        src/TypeChecker.cpp
        src/TypePrefix.cpp
)
   	
target_link_libraries(frontend PUBLIC modelica::model)
