add_library(serial uart_serial.cpp)

#fmt
find_package(fmt REQUIRED)
target_link_libraries(serial fmt::fmt)