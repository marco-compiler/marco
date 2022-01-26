set(SOURCES
		Utils.cpp
		Lexer.cpp
		Interval.cpp
		Range.cpp
		Shape.cpp
		LogMessage.cpp
		SourcePosition.cpp
		VariableFilter.cpp)

marco_add_library(utils SHARED
		${SOURCES}

		LINK_COMPONENTS
		Core
		Support)
