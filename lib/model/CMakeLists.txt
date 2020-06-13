modelicaAddLibrary(
	model SHARED 
	src/ModExp.cpp
   	src/ModErrors.cpp
   	src/Assigment.cpp
   	src/ModType.cpp
   	src/ModCall.cpp
   	src/ModLexerStateMachine.cpp
   	src/ModParser.cpp
   	src/Model.cpp
   	src/ModEquation.cpp
   	src/ModVariable.cpp
   	src/VectorAccess.cpp
   	src/ModConst.cpp
	src/ModMatchers.cpp)

target_link_libraries(model PUBLIC  modelica::utils)


