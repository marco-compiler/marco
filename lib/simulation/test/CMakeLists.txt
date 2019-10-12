include(GoogleTest)

add_executable(simulationTest src/SimulationTest.cpp)
add_executable(modelica::simulationTest ALIAS simulationTest) 
target_link_libraries(simulationTest PRIVATE gtest gtest_main modelica::lowerer)
target_include_directories(simulationTest PUBLIC include PRIVATE src)
target_compile_features(simulationTest PUBLIC cxx_std_17)

gtest_add_tests(TARGET      simulationTest
                TEST_SUFFIX .noArgs
                TEST_LIST   noArgsTests
)

