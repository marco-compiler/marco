include(GoogleTest)

add_executable(solverTest src/SolverTest.cpp)
add_executable(modelica::solverTest ALIAS solverTest) 
target_link_libraries(solverTest PRIVATE gtest gtest_main modelica::lowerer)
target_include_directories(solverTest PUBLIC include PRIVATE src)
target_compile_features(solverTest PUBLIC cxx_std_17)

gtest_add_tests(TARGET      solverTest
                TEST_SUFFIX .noArgs
                TEST_LIST   noArgsTests
)

