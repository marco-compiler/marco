include(GoogleTest)

ADD_EXECUTABLE(frontendTest src/LexerTest.cpp src/NumbersLexerTest.cpp src/LexerStateMachineTest.cpp src/ParserTest.cpp src/ParserErrorsTest.cpp src/ParserEquationTest.cpp src/VisitorTest.cpp src/ParserStatementTest.cpp src/ParserClassTest.cpp src/DumperTest.cpp)
ADD_EXECUTABLE(modelica::frontendTest ALIAS frontendTest) 
TARGET_LINK_LIBRARIES(frontendTest PRIVATE gtest gtest_main modelica::frontend)
TARGET_INCLUDE_DIRECTORIES(frontendTest PUBLIC include PRIVATE src)
TARGET_COMPILE_FEATURES(frontendTest PUBLIC cxx_std_17)

gtest_add_tests(TARGET      frontendTest
                TEST_SUFFIX .noArgs
                TEST_LIST   noArgsTests
)
