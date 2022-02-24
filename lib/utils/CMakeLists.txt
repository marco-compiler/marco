set(SOURCES
		Utils.cpp
		Lexer.cpp
		Interval.cpp
		Range.cpp
		LogMessage.cpp
		SourcePosition.cpp
		VariableFilter.cpp)

marco_add_library(utils STATIC
		${SOURCES}

		LINK_COMPONENTS
		Core
		Support)
