modelicaAddLibrary(
	frontend SHARED 
	src/Dumper.cpp 
	src/LexerStateMachine.cpp 
	src/Parser.cpp
   	src/ParserEquation.cpp
   	src/ParserErrors.cpp
   	src/Expr.cpp
   	src/Equation.cpp
   	src/ParserStatement.cpp
   	src/ParserClasses.cpp
   	src/Statement.cpp)
target_link_libraries(frontend PUBLIC modelica::model)

