modelicaAddLibrary(
	frontend SHARED 
	src/LexerStateMachine.cpp 
	src/Parser.cpp
	src/Call.cpp
	src/TypeChecker.cpp
	src/ConstantFolder.cpp
	src/Expression.cpp
   	src/ParserErrors.cpp)
   	
target_link_libraries(frontend PUBLIC modelica::model)

